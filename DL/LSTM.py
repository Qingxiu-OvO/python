import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
# [设置]: 强制使用非交互式后端 Agg (保存图片专用)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import tensorflow as tf
# [修改]: 引入 StandardScaler (对收益率这种正态分布数据，StandardScaler 比 MinMaxScaler 更好)
from sklearn.preprocessing import StandardScaler
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
BASE_SEED = 42
os.environ['PYTHONHASHSEED'] = str(BASE_SEED)

# --- 字体设置 ---
FONT_PROP = None
FONT_NAME = 'sans-serif'
CANDIDATE_FONTS = [
    "/System/Library/Fonts/STHeiti Light.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf"
]

try:
    found_font = False
    for path in CANDIDATE_FONTS:
        if os.path.exists(path):
            FONT_PROP = fm.FontProperties(fname=path, size=12)
            FONT_NAME = FONT_PROP.get_name()
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [FONT_NAME]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✅ 字体配置成功: {FONT_NAME}")
            found_font = True
            break
    if not found_font:
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"❌ 字体配置异常: {e}")

tf.keras.backend.clear_session()
plt.style.use('seaborn-v0_8') 

# ==============================================================================
# 1. 数据获取与特征工程 (关键修改：计算收益率)
# ==============================================================================
def get_and_prepare_data(ticker: str = '000001.SS') -> pd.DataFrame:
    print(f"正在下载 {ticker} 数据...")
    try:
        df = yf.download(ticker, start='2019-01-01', end=None, progress=False)
    except Exception as e:
        print(f"下载失败: {e}")
        sys.exit(1)
    
    if df.empty: sys.exit(1)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # --- 技术指标 (作为输入特征 X) ---
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 布林带
    df['BB_Upper'] = df['MA20'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_Lower'] = df['MA20'] - 2 * df['Close'].rolling(window=20).std()
    
    # --- [关键修改] 计算预测目标 (作为输出 Y) ---
    # 使用对数收益率: ln(Today / Yesterday)
    # 1. 收盘价收益率
    df['Log_Ret_Close'] = np.log(df['Close'] / df['Close'].shift(1))
    # 2. 最高价收益率 (定义为: 当日最高价 相对于 昨日最高价 的变化)
    # 注意：也可以定义为相对于昨日收盘价，这里保持逻辑一致性
    df['Log_Ret_High'] = np.log(df['High'] / df['High'].shift(1))

    # 清除计算产生的空值
    df = df.dropna()
    
    # 整理列顺序:
    # 前面是输入特征(X)，最后两列是预测目标(Y)
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                    'MA10', 'MA20', 'RSI', 'MACD', 'Signal', 'BB_Upper', 'BB_Lower',
                    'Log_Ret_High', 'Log_Ret_Close'] # <--- 目标在最后
    df = df[feature_cols]
    
    print(f"✅ 数据准备完成。特征数: {df.shape[1]-2}, 目标数: 2 (收益率)")
    return df

# ==============================================================================
# 2. 数据集处理
# ==============================================================================
def split_and_scale(df: pd.DataFrame, look_back: int) -> tuple[np.ndarray, np.ndarray, StandardScaler, pd.DataFrame]:
    # 切分时间
    train_df = df.loc['2020-01-01':'2024-12-31']
    test_df_raw = df.loc['2025-01-01':]

    if len(test_df_raw) == 0: sys.exit(1)

    # 拼接测试集所需的历史窗口
    full_dataset = pd.concat((train_df, test_df_raw), axis=0)
    test_inputs = full_dataset[len(full_dataset) - len(test_df_raw) - look_back:].values
    
    # [修改]: 使用 StandardScaler
    # 原因：收益率数据通常接近正态分布（钟形曲线），StandardScaler 比 MinMax 效果更好
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df.values)
    test_inputs_scaled = scaler.transform(test_inputs)
    
    return train_scaled, test_inputs_scaled, scaler, test_df_raw

def create_xy(dataset: np.ndarray, look_back: int) -> tuple[np.ndarray, np.ndarray]:
    X, Y = [], []
    # dataset 最后一列是 Close收益率(-1), 倒数第二列是 High收益率(-2)
    idx_high_ret = -2
    idx_close_ret = -1
    
    for i in range(look_back, len(dataset)):
        # 输入 X: 使用所有特征 (包括过去几天的收益率，这对预测很有帮助)
        X.append(dataset[i-look_back:i, :])
        
        # 输出 Y: 预测当天的 [High_Ret, Close_Ret]
        Y.append([dataset[i, idx_high_ret], dataset[i, idx_close_ret]]) 
        
    return np.array(X), np.array(Y)

# ==============================================================================
# 3. 模型构建
# ==============================================================================
def build_generic_lstm_model(layer_units: List[int], input_shape: Tuple[int, int]) -> Model:
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    for i, units in enumerate(layer_units):
        return_seq = (i < len(layer_units) - 1)
        model.add(LSTM(units=units, return_sequences=return_seq))
        model.add(Dropout(0.3))
    
    model.add(Dense(units=2)) # 输出层: 预测2个收益率值
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=0)

# ==============================================================================
# 4. 核心逻辑: 价格还原 (Reconstruction)
# ==============================================================================
def recover_prices(pred_returns_scaled: np.ndarray, scaler: StandardScaler, 
                   prev_prices_high: np.ndarray, prev_prices_close: np.ndarray, 
                   feature_total_count: int) -> tuple[np.ndarray, np.ndarray]:
    """
    将模型预测的【归一化收益率】还原为【真实价格点位】
    公式: 今日价格 = 昨日价格 * exp(今日预测对数收益率)
    """
    # 1. 反归一化 (Inverse Scale)
    # 构造填充矩阵，因为 scaler 是对所有列训练的
    dummy = np.zeros((len(pred_returns_scaled), feature_total_count))
    # 将预测值填入对应的收益率列位置 (最后两列)
    dummy[:, -2] = pred_returns_scaled[:, 0] # High Ret
    dummy[:, -1] = pred_returns_scaled[:, 1] # Close Ret
    
    # 反转 scaling
    res_unscaled = scaler.inverse_transform(dummy)
    
    # 提取真实的对数收益率
    pred_log_ret_high = res_unscaled[:, -2]
    pred_log_ret_close = res_unscaled[:, -1]
    
    # 2. 价格还原 (Price Reconstruction)
    # 预测价格 = 昨日价格 * exp(预测的对数收益率)
    rec_high = prev_prices_high * np.exp(pred_log_ret_high)
    rec_close = prev_prices_close * np.exp(pred_log_ret_close)
    
    return rec_high, rec_close

def evaluate_predictions(real: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    rmse = np.sqrt(mean_squared_error(real, pred))
    mae = mean_absolute_error(real, pred)
    return rmse, mae

# ==============================================================================
# 5. 主程序逻辑
# ==============================================================================
def main():
    LOOK_BACK = 30
    EPOCHS = 80
    BATCH_SIZE = 32
    N_ROUNDS = 5 # 5轮平均

    EXPERIMENTS = {
        "Exp1_Normal": [64],                 
        "Exp1_Wide":   [128],
        "Exp2_Normal": [128, 64],  
        "Exp2_Narrow": [64, 32],
        "Exp2_Wide": [256, 128],
        "Exp2_Same":   [128, 128],
    }

    # 1. 数据准备
    df = get_and_prepare_data()
    TOTAL_FEATURES = df.shape[1]
    
    train_scaled, test_inputs_scaled, scaler, test_df_target = split_and_scale(df, LOOK_BACK)
    X_train, y_train = create_xy(train_scaled, LOOK_BACK)
    X_test, y_test = create_xy(test_inputs_scaled, LOOK_BACK)

    # 真实价格 (用于评估)
    real_close = test_df_target['Close'].values
    real_high = test_df_target['High'].values
    dates = test_df_target.index
    
    # [关键]: 获取测试集每一天对应的"前一天价格" (用于从收益率还原价格)
    # test_df_target 是2025年的数据。
    # 它的第 i 天的基准价格，应该是第 i-1 天的价格。
    # 我们需要把整个序列向下移动一位，第一天的基准需要去历史数据里找（split时已处理连续性，但在pandas里操作更方便）
    
    # 获取包含2024最后一天的数据以便 shift
    full_target_prices = pd.concat([
        df.loc['2024-12-01':].iloc[-(len(test_df_raw)+1):], # 取足够长，只要最后len+1个
    ])
    # 实际上，test_df_target['Close'].shift(1) 会导致第一天是 NaN
    # 我们需要完整的价格序列来做 shift
    price_series_close = df['Close'].loc[test_df_target.index[0] : test_df_target.index[-1]]
    # 为了得到第一天的基准(昨日)，我们需要前一天的数据
    prev_date = df.index[df.index.get_loc(test_df_target.index[0]) - 1]
    
    # 构造"昨日价格"序列
    # 取出从 (测试集第一天前一天) 到 (测试集倒数第二天) 的价格
    ref_prices_high = df['High'].loc[prev_date : test_df_target.index[-2]].values
    ref_prices_close = df['Close'].loc[prev_date : test_df_target.index[-2]].values
    
    # 确保长度一致
    assert len(ref_prices_close) == len(real_close), "基准价格序列长度不匹配"

    final_results_summary = []

    print(f"\n======== 开始实验 (预测目标: 收益率 -> 还原为价格) ========")

    for exp_name, layers_config in EXPERIMENTS.items():
        print(f"\n>> [实验组]: {exp_name} 结构: {layers_config}")
        
        temp_maes = []
        temp_rmses = []
        temp_pred_high_list = []
        temp_pred_close_list = []

        for i in range(N_ROUNDS):
            print(f"   - 第 {i+1}/{N_ROUNDS} 次训练...", end="", flush=True)
            
            current_seed = BASE_SEED + i
            np.random.seed(current_seed)
            tf.random.set_seed(current_seed)
            tf.keras.backend.clear_session()
            
            model = build_generic_lstm_model(layers_config, (X_train.shape[1], X_train.shape[2]))
            model.fit(
                X_train, y_train, 
                epochs=EPOCHS, 
                batch_size=BATCH_SIZE, 
                verbose=0, 
                callbacks=[lr_schedule] 
            )
            
            # 1. 预测 (得到归一化的收益率)
            pred_rets_scaled = model.predict(X_test, verbose=0)
            
            # 2. 还原 (归一化收益率 -> 真实收益率 -> 真实价格)
            rec_high, rec_close = recover_prices(
                pred_rets_scaled, scaler, 
                ref_prices_high, ref_prices_close, 
                TOTAL_FEATURES
            )
            
            # 3. 评估 (对比还原后的价格 vs 真实价格)
            rmse, mae = evaluate_predictions(real_close, rec_close)
            
            temp_maes.append(mae)
            temp_rmses.append(rmse)
            temp_pred_high_list.append(rec_high)
            temp_pred_close_list.append(rec_close)
            
            print(f" 完成. (MAE: {mae:.2f})")

        # 计算平均
        avg_mae = np.mean(temp_maes)
        avg_rmse = np.mean(temp_rmses)
        avg_pred_high = np.mean(np.array(temp_pred_high_list), axis=0)
        avg_pred_close = np.mean(np.array(temp_pred_close_list), axis=0)
        
        print(f"   >> {exp_name} 平均 MAE: {avg_mae:.4f}")
        
        final_results_summary.append({
            "Experiment": exp_name,
            "Structure": str(layers_config),
            "Avg_MAE": avg_mae,
            "Avg_RMSE": avg_rmse,
            "Pred_High": avg_pred_high,
            "Pred_Close": avg_pred_close
        })

    results_df = pd.DataFrame(final_results_summary).sort_values(by="Avg_MAE")
    
    print("\n" + "="*60)
    print(f"最终实验报告 (按 {N_ROUNDS} 轮平均 MAE 排序)")
    print("="*60)
    print(results_df[["Experiment", "Structure", "Avg_MAE", "Avg_RMSE"]].to_string(index=False))
    
    best_exp = results_df.iloc[0]
    best_name = best_exp["Experiment"]
    print(f"\n🏆 最佳模型方案: {best_name} (平均MAE: {best_exp['Avg_MAE']:.4f})")

    # 绘图
    print(f"\n正在绘制最佳模型 ({best_name}) 的结果...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    ax1.set_title(f'上证 2025 最高价预测 (基于收益率预测还原, {best_name})', fontsize=14, fontproperties=FONT_PROP)
    ax1.plot(dates, real_high, label='实际最高价', color='#d62728', linewidth=2)
    ax1.plot(dates, best_exp["Pred_High"], label='预测最高价(还原后)', color='#1f77b4', linestyle='--', linewidth=1.5)
    ax1.legend(loc='upper left', prop=FONT_PROP)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel('价格', fontproperties=FONT_PROP)

    ax2.set_title(f'上证 2025 收盘价预测 (基于收益率预测还原, {best_name})', fontsize=14, fontproperties=FONT_PROP)
    ax2.plot(dates, real_close, label='实际收盘价', color='#2ca02c', linewidth=2)
    ax2.plot(dates, best_exp["Pred_Close"], label='预测收盘价(还原后)', color='#ff7f0e', linestyle='--', linewidth=1.5)
    ax2.legend(loc='upper left', prop=FONT_PROP)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel('价格', fontproperties=FONT_PROP)
    ax2.set_xlabel('日期', fontproperties=FONT_PROP)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    
    SAVE_NAME = f'LSTM_ReturnBased_Result_{best_name}.png'
    plt.savefig(SAVE_NAME, dpi=300)
    print(f"✅ 图表已保存至: {SAVE_NAME}")

if __name__ == "__main__":
    # 定义 test_df_raw 变量以修复引用范围问题 (helper fix)
    test_df_raw = [] 
    main()