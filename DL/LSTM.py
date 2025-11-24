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

# ==========================================
# 字体设置 (保留最终稳定版本)
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
# 1. 数据准备与特征工程 (优化: 更清晰的类型提示)
# ==========================================
def get_and_prepare_data(ticker: str = '000001.SS') -> pd.DataFrame:
    """获取数据并添加 MA10/MA20 特征"""
    print(f"正在下载 {ticker} 数据...")
    try:
        # 下载数据 (包含足够的历史数据以计算 MA20)
        df = yf.download(ticker, start='2019-10-01', end=None, progress=False)
    except Exception as e:
        print(f"下载失败: {e}")
        sys.exit(1)
    
    if df.empty:
        print("未获取到数据，请检查网络或股票代码。")
        sys.exit(1)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df[['Open', 'High', 'Low', 'Close']]
    
    # 特征工程
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df = df.dropna()
    print(f"✅ 数据处理完成。特征数量: {df.shape[1]}")
    return df

# ==========================================
# 2. 数据集切分与归一化 (核心防泄漏)
# ==========================================
def split_and_scale(df: pd.DataFrame, look_back: int) -> tuple[np.ndarray, np.ndarray, MinMaxScaler, pd.DataFrame]:
    """严格按时间切分并归一化，确保防泄漏"""
    train_df = df.loc['2020-01-01':'2024-12-31']
    test_df_raw = df.loc['2025-01-01':]

    full_dataset = pd.concat((train_df, test_df_raw), axis=0)
    # 截取测试集需要的输入部分 (包含重叠区)
    test_inputs = full_dataset[len(full_dataset) - len(test_df_raw) - look_back:].values
    
    # 归一化 (只 fit 训练集)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_df.values)
    test_inputs_scaled = scaler.transform(test_inputs)
    
    print(f"✅ 数据切分完成。训练样本: {len(train_df)}, 测试样本: {len(test_df_raw)}")
    return train_scaled, test_inputs_scaled, scaler, test_df_raw

def create_xy(dataset: np.ndarray, look_back: int) -> tuple[np.ndarray, np.ndarray]:
    """构造 LSTM 3D 数据格式 (样本数, 时间步, 特征数)"""
    X, Y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i-look_back:i, :])
        # 目标: High (索引 1), Close (索引 3)
        Y.append([dataset[i, 1], dataset[i, 3]]) 
    return np.array(X), np.array(Y)

# ==========================================
# 3. 模型构建与训练 (优化: 学习率调度器)
# ==========================================
def build_lstm_model(input_shape: tuple) -> Model:
    """构建 LSTM 模型"""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units=100, return_sequences=True), # 增加神经元数量
        Dropout(0.3), # 提高 Dropout
        LSTM(units=50, return_sequences=False),
        Dropout(0.3),
        Dense(units=2) # 输出 High, Close
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# 定义回调函数: 学习率调度器 (当 val_loss 停止改善时，降低学习率)
lr_schedule = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5,      # 学习率降低 50%
    patience=10,     # 10 个 epoch 没有改善就触发
    min_lr=0.00001,  # 最小学习率
    verbose=1
)

# ==========================================
# 4. 预测与评估 (优化: 反归一化函数独立，添加评估指标)
# ==========================================
def inverse_transform_helper(preds: np.ndarray, scaler: MinMaxScaler) -> tuple[np.ndarray, np.ndarray]:
    """将归一化的 [High, Close] 预测值反转为真实价格"""
    # 创建一个 (N, 6) 的零矩阵，因为 scaler 期望 6 列输入
    dummy = np.zeros((len(preds), 6))
    dummy[:, 1] = preds[:, 0] # 预测的 High 填入原始 High 列 (索引 1)
    dummy[:, 3] = preds[:, 1] # 预测的 Close 填入原始 Close 列 (索引 3)
    
    res = scaler.inverse_transform(dummy)
    return res[:, 1], res[:, 3] # 返回反转后的 High 和 Close

def evaluate_predictions(real: np.ndarray, pred: np.ndarray, name: str) -> None:
    """计算并打印评估指标"""
    rmse = np.sqrt(mean_squared_error(real, pred))
    mae = mean_absolute_error(real, pred)
    print(f"--- {name} 预测指标 (2025年) ---")
    print(f"均方根误差 (RMSE): {rmse:.2f}")
    print(f"平均绝对误差 (MAE): {mae:.2f}")
    print("---------------------------------")


# ==========================================
# 5. 主程序逻辑
# ==========================================
def main():
    LOOK_BACK = 30
    EPOCHS = 50
    BATCH_SIZE = 32

    # 1. 数据准备
    df = get_and_prepare_data()
    train_scaled, test_inputs_scaled, scaler, test_df_target = split_and_scale(df, LOOK_BACK)

    # 2. 构造 X, Y
    X_train, y_train = create_xy(train_scaled, LOOK_BACK)
    X_test, y_test = create_xy(test_inputs_scaled, LOOK_BACK)

    # 3. 训练模型
    model = build_lstm_model((LOOK_BACK, df.shape[1]))
    print("构建并训练 LSTM 模型...")
    model.fit(
        X_train, y_train, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        verbose=1, 
        validation_split=0.1,
        callbacks=[lr_schedule] # 加入学习率调度器
    )

    # 4. 预测与反归一化
    print("正在预测 2025 年走势...")
    predictions = model.predict(X_test)
    pred_high, pred_close = inverse_transform_helper(predictions, scaler)
    
    real_high = test_df_target['High'].values
    real_close = test_df_target['Close'].values
    dates = test_df_target.index

    # 5. 评估预测结果
    evaluate_predictions(real_close, pred_close, "收盘价")
    evaluate_predictions(real_high, pred_high, "最高价")

    # 6. 绘图 (确保所有中文标签都使用 FONT_PROP)
    print("正在生成图表...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # 上图：最高价对比 
    ax1.set_title('上证指数 2025年 最高价预测 vs 实际', 
                  fontsize=14, fontproperties=FONT_PROP)
    ax1.plot(dates, real_high, label='实际最高价', color='#d62728', linewidth=2)
    ax1.plot(dates, pred_high, label='预测最高价', color='#1f77b4', linestyle='--', linewidth=1.5)
    ax1.legend(loc='upper left', prop=FONT_PROP)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel('价格', fontproperties=FONT_PROP)

    # 下图：收盘价对比
    ax2.set_title('上证指数 2025年 收盘价预测 vs 实际', 
                  fontsize=14, fontproperties=FONT_PROP)
    ax2.plot(dates, real_close, label='实际收盘价', color='#2ca02c', linewidth=2)
    ax2.plot(dates, pred_close, label='预测收盘价', color='#ff7f0e', linestyle='--', linewidth=1.5)
    ax2.legend(loc='upper left', prop=FONT_PROP)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel('价格', fontproperties=FONT_PROP)
    ax2.set_xlabel('日期', fontproperties=FONT_PROP)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()

    plt.tight_layout()
    plt.show() 

if __name__ == "__main__":
    main()