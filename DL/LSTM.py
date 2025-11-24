import numpy as np
import pandas as pd
import yfinance as yf
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

# ==========================================
# 字体设置 (保留你的稳定配置)
# ==========================================
FONT_PROP = None
FONT_NAME = 'Arial Unicode MS' 
try:
    # 强制使用 TkAgg 后端
    import matplotlib
    matplotlib.use('TkAgg') 
    
    # 尝试注册字体
    CHINESE_FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
    if os.path.exists(CHINESE_FONT_PATH):
        FONT_PROP = fm.FontProperties(fname=CHINESE_FONT_PATH, size=12)
        plt.rcParams['font.sans-serif'] = [FONT_PROP.get_name()]
        plt.rcParams['axes.unicode_minus'] = False
        print(f"✅ 字体成功设置: {FONT_PROP.get_name()}")

except Exception as e:
    print(f"❌ 字体配置失败，图表可能乱码。错误: {e}")
    FONT_PROP = None

# 清理 TF 内存状态并设置绘图风格
tf.keras.backend.clear_session()
plt.style.use('seaborn-v0_8') 

# ==========================================
# 1. 数据准备与特征工程
# ==========================================
def get_and_prepare_data(ticker: str = '000001.SS') -> pd.DataFrame:
    """获取数据并添加 MA10/MA20 特征"""
    print(f"正在下载 {ticker} 数据...")
    try:
        # 下载数据
        df = yf.download(ticker, start='2019-10-01', end=None, progress=False)
    except Exception as e:
        print(f"下载失败: {e}")
        sys.exit(1)
    
    if df.empty:
        print("未获取到数据，请检查网络或股票代码。")
        sys.exit(1)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # 特征工程
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    # [新增情绪特征 1]: 成交量相对均值的比率 (V_Ratio)
    df['V_MA30'] = df['Volume'].rolling(window=30).mean()
    df['V_Ratio'] = df['Volume'] / df['V_MA30']
    
    # [新增情绪特征 2]: 历史波动率 (Historical Volatility, 20日)
    # 计算日对数收益率
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    # 20日历史波动率 (年化，乘以sqrt(252))
    df['HV_20'] = df['Log_Return'].rolling(window=20).std() * np.sqrt(252)
    df = df.dropna()
     # [修改点 2]: 最终特征列表 (移除 V_MA30 和 Log_Return)
    df = df[['Open', 'High', 'Low', 'Close', 'MA10', 'MA20', 'V_Ratio', 'HV_20']]
    print(f"✅ 数据处理完成。特征数量: {df.shape[1]}")
    return df

# ==========================================
# 2. 数据集切分与归一化
# ==========================================
def split_and_scale(df: pd.DataFrame, look_back: int) -> tuple[np.ndarray, np.ndarray, MinMaxScaler, pd.DataFrame]:
    """严格按时间切分并归一化"""
    train_df = df.loc['2020-01-01':'2024-12-31']
    test_df_raw = df.loc['2025-01-01':]

    full_dataset = pd.concat((train_df, test_df_raw), axis=0)
    test_inputs = full_dataset[len(full_dataset) - len(test_df_raw) - look_back:].values
    
    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_df.values)
    test_inputs_scaled = scaler.transform(test_inputs)
    
    print(f"✅ 数据切分完成。训练样本: {len(train_df)}, 测试样本: {len(test_df_raw)}")
    return train_scaled, test_inputs_scaled, scaler, test_df_raw

def create_xy(dataset: np.ndarray, look_back: int) -> tuple[np.ndarray, np.ndarray]:
    """构造 LSTM 3D 数据格式"""
    X, Y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i-look_back:i, :])
        Y.append([dataset[i, 1], dataset[i, 3]]) 
    return np.array(X), np.array(Y)

# ==========================================
# 3. 动态模型构建 (修改点: 支持不同层数)
# ==========================================
def build_generic_lstm_model(layer_units: List[int], input_shape: Tuple[int, int]) -> Model:
    """
    根据传入的单元列表动态构建 LSTM 模型。
    例如 layer_units=[100, 50] 构建两层，[128] 构建一层。
    """
    model = Sequential()
    
    for i, units in enumerate(layer_units):
        # 逻辑：如果是最后一层 LSTM，return_sequences 必须为 False
        # 如果后面还有 LSTM 层，return_sequences 必须为 True
        is_last_lstm_layer = (i == len(layer_units) - 1)
        return_seq = not is_last_lstm_layer
        
        if i == 0:
            # 第一层必须指定 input_shape
            model.add(LSTM(units=units, return_sequences=return_seq, input_shape=input_shape))
        else:
            # 后续层自动推断
            model.add(LSTM(units=units, return_sequences=return_seq))
            
        model.add(Dropout(0.3))
    
    model.add(Dense(units=2)) # 输出 High, Close
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# 学习率调度器
lr_schedule = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5,      
    patience=10,     
    min_lr=0.00001,  
    verbose=0 # 实验时静默
)

# ==========================================
# 4. 预测与评估辅助函数
# ==========================================
def inverse_transform_helper(preds: np.ndarray, scaler: MinMaxScaler) -> tuple[np.ndarray, np.ndarray]:
    dummy = np.zeros((len(preds), 8))
    dummy[:, 1] = preds[:, 0]
    dummy[:, 3] = preds[:, 1]
    res = scaler.inverse_transform(dummy)
    return res[:, 1], res[:, 3]

def evaluate_predictions(real: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    rmse = np.sqrt(mean_squared_error(real, pred))
    mae = mean_absolute_error(real, pred)
    return rmse, mae

# ==========================================
# 5. 主程序逻辑 (修改点: 实验循环)
# ==========================================
def main():
    LOOK_BACK = 30
    EPOCHS = 50
    BATCH_SIZE = 32

    # --- 定义实验配置 ---
    # 键是实验名称，值是 LSTM 层结构的列表
    # 例如 [128, 64] 代表第一层128个单元，第二层64个单元
    EXPERIMENTS = {
        "Exp1_Single_Layer": [64],                 # 单层
        "Exp1_Single_Layer_Narrow": [32],          # 单层 (窄)
        "Exp1_Single_Layer_Wide": [128],           # 单层 (更宽)
        "Exp2_Two_Layers":   [128, 64],            # 双层 (基准)
        "Exp2_Two_Layers_Narrow":   [128, 64],     # 双层 (窄)
        "Exp2_Two_Layers_Wide":   [256, 128],      # 双层 (更宽)
        "Exp3_Three_Layers": [128, 64, 32],        # 三层 (深层)
        "Exp3_Three_Layers_Narrow": [64, 32, 16],  # 三层 (窄)
        "Exp3_Three_Layers_Wide": [256, 128, 64],  # 三层 (宽)
    }

    # 1. 数据准备
    df = get_and_prepare_data()
    train_scaled, test_inputs_scaled, scaler, test_df_target = split_and_scale(df, LOOK_BACK)
    X_train, y_train = create_xy(train_scaled, LOOK_BACK)
    X_test, y_test = create_xy(test_inputs_scaled, LOOK_BACK)

    real_close = test_df_target['Close'].values
    real_high = test_df_target['High'].values
    dates = test_df_target.index

    results_data = [] # 用于存储结果

    print(f"\n======== 开始不同隐藏层数量的对比实验 ========")

    # 2. 循环实验
    for exp_name, layers_config in EXPERIMENTS.items():
        print(f"\n>> 正在训练模型: {exp_name} (结构: {layers_config})...")
        
        # 清理内存
        tf.keras.backend.clear_session()
        
        # 构建模型
        model = build_generic_lstm_model(layers_config, (X_train.shape[1], X_train.shape[2]))
        
        # 训练 (verbose=0 不刷屏，只显示结果)
        history = model.fit(
            X_train, y_train, 
            epochs=EPOCHS, 
            batch_size=BATCH_SIZE, 
            verbose=0, 
            validation_split=0.1,
            callbacks=[lr_schedule]
        )
        
        # 预测
        preds = model.predict(X_test, verbose=0)
        p_high, p_close = inverse_transform_helper(preds, scaler)
        
        # 评估
        rmse, mae = evaluate_predictions(real_close, p_close)
        val_loss = history.history['val_loss'][-1]
        
        print(f"   [完成] MAE(收盘价): {mae:.4f} | RMSE: {rmse:.4f} | Val_Loss: {val_loss:.6f}")
        
        results_data.append({
            "Experiment": exp_name,
            "Structure": str(layers_config),
            "Layers_Count": len(layers_config),
            "MAE": mae,
            "RMSE": rmse,
            "Val_Loss": val_loss,
            "Pred_High": p_high,   # 暂存预测结果以便画图
            "Pred_Close": p_close
        })

    # 3. 结果总结
    results_df = pd.DataFrame(results_data).sort_values(by="MAE")
    print("\n" + "="*50)
    print("实验结果汇总 (按 MAE 误差从小到大排序)")
    print("="*50)
    print(results_df[["Experiment", "Structure", "MAE", "RMSE", "Val_Loss"]].to_string(index=False))
    
    # 获取最佳模型的数据
    best_exp = results_df.iloc[0]
    best_name = best_exp["Experiment"]
    print(f"\n🏆 最佳模型是: {best_name} (MAE: {best_exp['MAE']:.4f})")

    # 4. 绘图 (只绘制最佳模型的效果)
    print(f"正在绘制最佳模型 ({best_name}) 的图表...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # 上图：最高价
    ax1.set_title(f'上证指数 2025年 最高价预测 (最佳模型: {best_name})', fontsize=14, fontproperties=FONT_PROP)
    ax1.plot(dates, real_high, label='实际最高价', color='#d62728', linewidth=2)
    ax1.plot(dates, best_exp["Pred_High"], label='预测最高价', color='#1f77b4', linestyle='--', linewidth=1.5)
    ax1.legend(loc='upper left', prop=FONT_PROP)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel('价格', fontproperties=FONT_PROP)

    # 下图：收盘价
    ax2.set_title(f'上证指数 2025年 收盘价预测 (最佳模型: {best_name})', fontsize=14, fontproperties=FONT_PROP)
    ax2.plot(dates, real_close, label='实际收盘价', color='#2ca02c', linewidth=2)
    ax2.plot(dates, best_exp["Pred_Close"], label='预测收盘价', color='#ff7f0e', linestyle='--', linewidth=1.5)
    ax2.legend(loc='upper left', prop=FONT_PROP)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel('价格', fontproperties=FONT_PROP)
    ax2.set_xlabel('日期', fontproperties=FONT_PROP)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()

    plt.tight_layout()
    
    # 保存图片
    SAVE_NAME = f'LSTM_Compare_Best_{best_name}.png'
    plt.savefig(SAVE_NAME, dpi=300)
    print(f"✅ 图表已保存至: {SAVE_NAME}")
    
    # 弹窗显示 (可选)
    # plt.show()

if __name__ == "__main__":
    main()