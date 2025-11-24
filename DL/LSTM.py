import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
# [关键设置]: 强制使用非交互式后端 Agg (保存图片专用，防止在无头服务器或Mac上弹窗报错)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import sys
import os
from typing import List, Tuple

# ==============================================================================
# 0. 基础设置 (随机种子 & 字体配置)
# ==============================================================================

# 基础随机种子 (后续会在每轮训练中微调这个种子以保证独立性)
BASE_SEED = 42
os.environ['PYTHONHASHSEED'] = str(BASE_SEED)

# --- 字体设置 (Mac/Linux/Windows 兼容性处理) ---
# 目的：解决 Matplotlib 无法显示中文，显示为方框的问题
FONT_PROP = None
FONT_NAME = 'sans-serif'

# Mac 系统常用 .ttf 中文字体路径 (优先级从高到低)
# 优先使用 .ttf 文件，避免 .ttc 集合文件导致的底层读取错误
CANDIDATE_FONTS = [
    "/System/Library/Fonts/STHeiti Light.ttf",             # 华文黑体 (Mac最稳)
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf", # 通用 Unicode
    "/Library/Fonts/Arial Unicode.ttf"
]

try:
    found_font = False
    # 遍历候选列表，找到第一个存在的字体文件
    for path in CANDIDATE_FONTS:
        if os.path.exists(path):
            FONT_PROP = fm.FontProperties(fname=path, size=12)
            FONT_NAME = FONT_PROP.get_name()
            
            # 设置 Matplotlib 全局参数
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [FONT_NAME]
            plt.rcParams['axes.unicode_minus'] = False # 解决负号显示为方块的问题
            
            print(f"✅ 字体配置成功: {FONT_NAME} (路径: {path})")
            found_font = True
            break
    
    # 如果找不到特定路径，回退到系统名称查找
    if not found_font:
        print("⚠️ 未找到预设路径字体，尝试系统自动回退...")
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

except Exception as e:
    print(f"❌ 字体配置异常: {e}")

# 清理 TensorFlow 之前的会话，释放显存/内存
tf.keras.backend.clear_session()
# 设置绘图风格
plt.style.use('seaborn-v0_8') 

# ==============================================================================
# 1. 数据获取与特征工程
# ==============================================================================
def get_and_prepare_data(ticker: str = '000001.SS') -> pd.DataFrame:
    """
    从 Yahoo Finance 下载数据，并计算技术指标作为特征。
    """
    print(f"正在下载 {ticker} 数据...")
    try:
        # 下载足够长的时间跨度以确保计算均线时不产生空值
        df = yf.download(ticker, start='2019-10-01', end=None, progress=False)
    except Exception as e:
        print(f"下载失败: {e}")
        sys.exit(1)
    
    if df.empty:
        print("未获取到数据，请检查网络。")
        sys.exit(1)

    # 处理多级索引问题 (yfinance 新版特性)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    # 保留基础 OHLCV 数据
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # --- 特征工程 (Feature Engineering) ---
    # 1. 移动平均线 (Trend)
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # 2. RSI (相对强弱指数 - Momentum)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 3. MACD (趋势指标)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 4. 布林带 (波动率指标)
    df['BB_Upper'] = df['MA20'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_Lower'] = df['MA20'] - 2 * df['Close'].rolling(window=20).std()
    
    # 删除因计算指标产生的 NaN 行 (前20-30行)
    df = df.dropna()
    
    # 最终选用的特征列
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                    'MA10', 'MA20', 'RSI', 'MACD', 'Signal', 'BB_Upper', 'BB_Lower']
    df = df[feature_cols]
    
    print(f"✅ 特征工程完成。当前特征数: {df.shape[1]}")
    return df

# ==============================================================================
# 2. 数据集处理 (切分、归一化、时间窗构建)
# ==============================================================================
def split_and_scale(df: pd.DataFrame, look_back: int) -> tuple[np.ndarray, np.ndarray, MinMaxScaler, pd.DataFrame]:
    """
    将数据按时间严格切分为训练集(2020-2024)和测试集(2025)，并进行归一化。
    注意：Scaler 只能在训练集上 fit，防止数据泄漏。
    """
    train_df = df.loc['2020-01-01':'2024-12-31']
    test_df_raw = df.loc['2025-01-01':]

    if len(test_df_raw) == 0:
        print("错误: 2025年数据为空，无法进行测试。")
        sys.exit(1)

    # 为了让测试集第一天有足够的历史数据，需要拼接训练集末尾的数据
    full_dataset = pd.concat((train_df, test_df_raw), axis=0)
    test_inputs = full_dataset[len(full_dataset) - len(test_df_raw) - look_back:].values
    
    # 归一化 (0~1之间)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_df.values) # 只在训练集上学习参数
    test_inputs_scaled = scaler.transform(test_inputs)   # 测试集应用相同参数
    
    return train_scaled, test_inputs_scaled, scaler, test_df_raw

def create_xy(dataset: np.ndarray, look_back: int) -> tuple[np.ndarray, np.ndarray]:
    """
    将时间序列数据转换为 LSTM 需要的监督学习格式 (X, Y)。
    输入: 过去 look_back 天的所有特征
    输出: 当天的 High 和 Close
    """
    X, Y = [], []
    for i in range(look_back, len(dataset)):
        # X: 从 i-look_back 到 i-1 的数据
        X.append(dataset[i-look_back:i, :])
        # Y: 第 i 天的数据 (High在索引1, Close在索引3)
        Y.append([dataset[i, 1], dataset[i, 3]]) 
    return np.array(X), np.array(Y)

# ==============================================================================
# 3. 模型构建与预测辅助
# ==============================================================================
def build_generic_lstm_model(layer_units: List[int], input_shape: Tuple[int, int]) -> Model:
    """
    动态构建 LSTM 模型。
    layer_units: 列表，例如 [64, 32] 表示两层 LSTM，节点数分别为 64 和 32。
    """
    model = Sequential()
    for i, units in enumerate(layer_units):
        # 如果不是最后一层 LSTM，必须设置 return_sequences=True 以传递序列给下一层
        return_seq = (i < len(layer_units) - 1)
        
        if i == 0:
            # 第一层需要指定输入形状
            model.add(LSTM(units=units, return_sequences=return_seq, input_shape=input_shape))
        else:
            model.add(LSTM(units=units, return_sequences=return_seq))
        
        # Dropout 防止过拟合
        model.add(Dropout(0.3))
    
    # 输出层: 预测 High 和 Close 两个值
    model.add(Dense(units=2))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# 学习率调度器: 当 loss 不再下降时，自动减小学习率
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=0)

def inverse_transform_helper(preds: np.ndarray, scaler: MinMaxScaler, feature_count: int) -> tuple[np.ndarray, np.ndarray]:
    """
    反归一化工具。因为 Scaler 是针对所有特征训练的，所以需要构建一个填充矩阵来还原。
    """
    dummy = np.zeros((len(preds), feature_count))
    # 将预测值填回 High (idx 1) 和 Close (idx 3) 的位置
    dummy[:, 1] = preds[:, 0]
    dummy[:, 3] = preds[:, 1]
    res = scaler.inverse_transform(dummy)
    return res[:, 1], res[:, 3]

def evaluate_predictions(real: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """计算均方根误差 (RMSE) 和 平均绝对误差 (MAE)"""
    rmse = np.sqrt(mean_squared_error(real, pred))
    mae = mean_absolute_error(real, pred)
    return rmse, mae

def get_model_predictions(model: Model, X_data: np.ndarray, feature_count: int, scaler: MinMaxScaler) -> tuple[np.ndarray, np.ndarray]:
    """封装预测和反归一化过程"""
    preds = model.predict(X_data, verbose=0)
    return inverse_transform_helper(preds, scaler, feature_count)

# ==============================================================================
# 4. 主程序逻辑 (多轮实验平均版)
# ==============================================================================
def main():
    # --- 核心参数 ---
    LOOK_BACK = 30
    EPOCHS = 100 
    BATCH_SIZE = 32
    N_ROUNDS = 5  # [新增] 每个实验模型运行的次数，用于取平均值

    # --- 实验配置字典 ---
    # Key: 实验名称
    # Value: LSTM 层结构列表
    EXPERIMENTS = {
        "Exp1_Normal": [64],                 
        "Exp1_Narrow": [32],
        "Exp1_Wide":   [128],
        "Exp2_Narrow": [128, 64],  
        "Exp2_Wide":   [256, 128],
        "Exp2_Small":  [64, 32],          
        "Exp3_Deep":   [128, 64, 32],
    }

    # 1. 准备数据
    df = get_and_prepare_data()
    NUM_FEATURES = df.shape[1] 
    
    train_scaled, test_inputs_scaled, scaler, test_df_target = split_and_scale(df, LOOK_BACK)
    X_train, y_train = create_xy(train_scaled, LOOK_BACK)
    X_test, y_test = create_xy(test_inputs_scaled, LOOK_BACK)

    real_close = test_df_target['Close'].values
    real_high = test_df_target['High'].values
    dates = test_df_target.index

    final_results_summary = [] # 存储所有实验的最终平均结果

    print(f"\n======== 开始实验 (每个模型运行 {N_ROUNDS} 轮取平均值) ========")

    # 2. 外层循环：遍历不同的模型结构
    for exp_name, layers_config in EXPERIMENTS.items():
        print(f"\n>> [实验组]: {exp_name} 结构: {layers_config}")
        
        # 用于存储 N_ROUNDS 次运行的临时数据
        temp_maes = []
        temp_rmses = []
        # 存储每次预测的原始价格数组，最后求平均曲线
        temp_pred_high_list = [] 
        temp_pred_close_list = []

        # 3. 内层循环：每个模型跑 N_ROUNDS 次
        for i in range(N_ROUNDS):
            print(f"   - 第 {i+1}/{N_ROUNDS} 次训练...", end="", flush=True)
            
            # [关键] 每次设置不同的种子，确保初始权重不同
            current_seed = BASE_SEED + i
            np.random.seed(current_seed)
            tf.random.set_seed(current_seed)
            tf.keras.backend.clear_session() # 清理内存
            
            # 构建并训练模型
            model = build_generic_lstm_model(layers_config, (X_train.shape[1], X_train.shape[2]))
            model.fit(
                X_train, y_train, 
                epochs=EPOCHS, 
                batch_size=BATCH_SIZE, 
                verbose=0, 
                validation_split=0.1,
                callbacks=[lr_schedule] 
            )
            
            # 预测
            p_high, p_close = get_model_predictions(model, X_test, NUM_FEATURES, scaler)
            rmse, mae = evaluate_predictions(real_close, p_close)
            
            # 记录单次结果
            temp_maes.append(mae)
            temp_rmses.append(rmse)
            temp_pred_high_list.append(p_high)
            temp_pred_close_list.append(p_close)
            
            print(f" 完成. (MAE: {mae:.2f})")

        # 4. 计算平均结果 (Ensemble Averaging)
        avg_mae = np.mean(temp_maes)
        avg_rmse = np.mean(temp_rmses)
        # 将5次预测的曲线（数组）在垂直方向取平均，得到一条更平滑的曲线
        avg_pred_high = np.mean(np.array(temp_pred_high_list), axis=0)
        avg_pred_close = np.mean(np.array(temp_pred_close_list), axis=0)
        
        print(f"   >> {exp_name} 平均 MAE: {avg_mae:.4f} | 平均 RMSE: {avg_rmse:.4f}")
        
        final_results_summary.append({
            "Experiment": exp_name,
            "Structure": str(layers_config),
            "Avg_MAE": avg_mae,
            "Avg_RMSE": avg_rmse,
            "Pred_High": avg_pred_high,   # 存储平均预测曲线
            "Pred_Close": avg_pred_close
        })

    # 5. 结果排序与展示
    results_df = pd.DataFrame(final_results_summary).sort_values(by="Avg_MAE")
    
    print("\n" + "="*60)
    print(f"最终实验报告 (按 {N_ROUNDS} 轮平均 MAE 排序)")
    print("="*60)
    print(results_df[["Experiment", "Structure", "Avg_MAE", "Avg_RMSE"]].to_string(index=False))
    
    # 获取最佳模型数据
    best_exp = results_df.iloc[0]
    best_name = best_exp["Experiment"]
    print(f"\n🏆 最佳模型方案: {best_name} (平均MAE: {best_exp['Avg_MAE']:.4f})")

    # 6. 绘图 (绘制最佳模型的平均预测结果)
    print(f"\n正在绘制最佳模型 ({best_name}) 的平均预测图表...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # 绘制最高价
    ax1.set_title(f'上证 2025 最高价预测 ({best_name}, {N_ROUNDS}轮平均)', fontsize=14, fontproperties=FONT_PROP)
    ax1.plot(dates, real_high, label='实际最高价', color='#d62728', linewidth=2)
    ax1.plot(dates, best_exp["Pred_High"], label='预测最高价(平均)', color='#1f77b4', linestyle='--', linewidth=1.5)
    ax1.legend(loc='upper left', prop=FONT_PROP)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel('价格', fontproperties=FONT_PROP)

    # 绘制收盘价
    ax2.set_title(f'上证 2025 收盘价预测 ({best_name}, {N_ROUNDS}轮平均)', fontsize=14, fontproperties=FONT_PROP)
    ax2.plot(dates, real_close, label='实际收盘价', color='#2ca02c', linewidth=2)
    ax2.plot(dates, best_exp["Pred_Close"], label='预测收盘价(平均)', color='#ff7f0e', linestyle='--', linewidth=1.5)
    ax2.legend(loc='upper left', prop=FONT_PROP)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel('价格', fontproperties=FONT_PROP)
    ax2.set_xlabel('日期', fontproperties=FONT_PROP)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    
    SAVE_NAME = f'LSTM_Final_Result_{best_name}.png'
    plt.savefig(SAVE_NAME, dpi=300)
    print(f"✅ 图表已保存至: {SAVE_NAME}")

if __name__ == "__main__":
    main()