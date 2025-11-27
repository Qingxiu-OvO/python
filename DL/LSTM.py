import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import sys
import os
import random
from typing import List, Tuple

# ==============================================================================
# 0. 基础设置
# ==============================================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀以此设备运行: MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("🚀以此设备运行: CUDA (Nvidia GPU)")
else:
    device = torch.device("cpu")
    print("⚠️以此设备运行: CPU")

BASE_SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(BASE_SEED)

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
            found_font = True
            break
except Exception:
    pass
plt.style.use('seaborn-v0_8') 

# ==============================================================================
# 1. 数据获取与特征工程 (核心修改：纯相对值 + 阈值过滤)
# ==============================================================================
def get_and_prepare_data(ticker: str = '000001.SS') -> pd.DataFrame:
    print(f"正在下载 {ticker} 数据...")
    try:
        # [修改]: 使用 2015 至今的数据
        df = yf.download(ticker, start='2015-01-01', end=None, progress=False)
    except Exception as e:
        print(f"下载失败: {e}")
        sys.exit(1)
    
    if df.empty: sys.exit(1)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # --- 特征工程 (保持相对值逻辑) ---
    df['Log_Ret_Close'] = np.log(df['Close'] / df['Close'].shift(1) + 1e-8)
    df['Log_Ret_Open']  = np.log(df['Open'] / df['Open'].shift(1) + 1e-8)
    df['Log_Ret_High']  = np.log(df['High'] / df['High'].shift(1) + 1e-8)
    df['Log_Ret_Low']   = np.log(df['Low'] / df['Low'].shift(1) + 1e-8)
    df['Log_Ret_Vol']   = np.log(df['Volume'] / df['Volume'].shift(1).replace(0, 1))

    ma10 = df['Close'].rolling(window=10).mean()
    df['MA10_Bias'] = (df['Close'] - ma10) / ma10
    ma20 = df['Close'].rolling(window=20).mean()
    df['MA20_Bias'] = (df['Close'] - ma20) / ma20
    ma60 = df['Close'].rolling(window=60).mean()
    df['MA60_Bias'] = (df['Close'] - ma60) / ma60

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd_raw = ema12 - ema26
    signal_raw = macd_raw.ewm(span=9, adjust=False).mean()
    df['MACD_Norm'] = macd_raw / df['Close']
    df['Signal_Norm'] = signal_raw / df['Close']

    rolling_mean = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    upper = rolling_mean + 2 * rolling_std
    lower = rolling_mean - 2 * rolling_std
    df['BB_PctB'] = (df['Close'] - lower) / (upper - lower)
    df['BB_Width'] = (upper - lower) / rolling_mean

    # --- 预测目标 ---
    df['Price_Change'] = df['Close'].pct_change().shift(-1)
    
    # 阈值过滤
    THRESHOLD = 0.002
    df_filtered = df[abs(df['Price_Change']) > THRESHOLD].copy()
    df_filtered['Target_Direction'] = (df_filtered['Price_Change'] > 0).astype(np.float32)

    df_filtered = df_filtered.dropna()
    
    # [关键修改]: 我们把 Price_Change 也加进 feature_cols，但放在 Target 后面
    # 这样它不会影响训练 (我们会在 split 时把它切掉)，但方便后续回测提取
    feature_cols = [
        'Log_Ret_Close', 'Log_Ret_Open', 'Log_Ret_High', 'Log_Ret_Low', 'Log_Ret_Vol',
        'MA10_Bias', 'MA20_Bias', 'MA60_Bias',
        'RSI', 'MACD_Norm', 'Signal_Norm',
        'BB_PctB', 'BB_Width',
        'Target_Direction', # 倒数第二列：标签
        'Price_Change'      # 最后一列：真实涨跌幅 (用于回测，不用于训练)
    ]
    
    final_df = df_filtered[feature_cols]
    
    print(f"✅ 数据准备完成。包含回测数据列。")
    return final_df

# ==============================================================================
# 2. 数据集处理
# ==============================================================================
def split_and_scale(df: pd.DataFrame, look_back: int) -> tuple:
    train_df = df.loc[:'2024-12-31'] # 动态切分
    test_df_raw = df.loc['2025-01-01':]

    if len(train_df) == 0 or len(test_df_raw) == 0:
        sys.exit(1)

    # 拼接历史窗口
    full_dataset = pd.concat((train_df, test_df_raw), axis=0)
    # 这里的 .values 包含了所有列，包括 Target 和 Price_Change
    test_inputs = full_dataset[len(full_dataset) - len(test_df_raw) - look_back:].values
    
    scaler = StandardScaler()
    
    # [关键修改]: 训练集缩放
    # 我们只缩放特征列 (即排除最后两列 Target_Direction 和 Price_Change)
    # iloc[:, :-2] 取除了最后两列之外的所有列
    X_train_scaled = scaler.fit_transform(train_df.iloc[:, :-2].values)
    
    # y_train 取倒数第二列 (Target_Direction)
    y_train = train_df.iloc[:, -2].values.reshape(-1, 1)

    # 测试集缩放 (同样只缩放特征列)
    X_test_inputs_scaled = scaler.transform(test_inputs[:, :-2])
    
    # 拼回去：缩放后的X + 原始Target + 原始Price_Change
    test_inputs_scaled = np.hstack([
        X_test_inputs_scaled, 
        test_inputs[:, -2].reshape(-1, 1), # Target
        test_inputs[:, -1].reshape(-1, 1)  # Price_Change
    ])

    return X_train_scaled, y_train, test_inputs_scaled, scaler, test_df_raw

def create_xy(X_data: np.ndarray, y_data: np.ndarray, look_back: int):
    X, Y = [], []
    for i in range(look_back, len(X_data)):
        X.append(X_data[i-look_back:i, :])
        Y.append([y_data[i, 0]]) 
    return np.array(X), np.array(Y)

# ==============================================================================
# 3. 模型构建
# ==============================================================================
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size=1, dropout_prob=0.3):
        super(LSTMClassifier, self).__init__()
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        prev_size = input_size
        for hidden_size in hidden_layer_sizes:
            self.layers.append(nn.LSTM(prev_size, hidden_size, batch_first=True))
            self.dropouts.append(nn.Dropout(dropout_prob))
            prev_size = hidden_size
            
        self.fc = nn.Linear(prev_size, output_size)

    def forward(self, x):
        out = x
        for i in range(len(self.layers)):
            out, _ = self.layers[i](out)
            out = self.dropouts[i](out)
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

# ==============================================================================
# 4. 评估逻辑
# ==============================================================================
def evaluate_predictions(real_labels: np.ndarray, pred_logits: np.ndarray) -> float:
    pred_probs = 1.0 / (1.0 + np.exp(-pred_logits))
    pred_labels = (pred_probs >= 0.5).astype(int)
    accuracy = accuracy_score(real_labels, pred_labels)
    return accuracy

def run_backtest(model, X_test_tensor, test_returns, dates):
    """
    回测函数
    model: 训练好的模型
    X_test_tensor: 测试集输入特征
    test_returns: 测试集每天的真实涨跌幅 (Price_Change)
    dates: 测试集日期
    """
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        probs = 1.0 / (1.0 + np.exp(-logits.cpu().numpy()))
        # 生成信号：1=买入/持有，0=空仓/卖出
        signals = (probs >= 0.5).astype(int).flatten()
    
    # 计算策略收益
    # 策略逻辑：如果预测涨(1)，则获得当天的 Price_Change；如果预测跌(0)，收益为0
    strategy_returns = signals * test_returns
    
    # 计算资金曲线 (Cumulative Returns)
    # 初始资金设为 1
    cumulative_market = np.cumprod(1 + test_returns)
    cumulative_strategy = np.cumprod(1 + strategy_returns)
    
    # 计算最终收益率
    total_return_market = cumulative_market[-1] - 1
    total_return_strategy = cumulative_strategy[-1] - 1
    
    print("\n" + "="*40)
    print("💰 回测报告 (Backtest Report)")
    print(f"市场基准收益率: {total_return_market:.2%}")
    print(f"LSTM 策略收益率: {total_return_strategy:.2%}")
    if total_return_strategy > total_return_market:
        print("🎉 恭喜！策略跑赢了市场！")
    else:
        print("🥀 遗憾，策略没跑赢市场。")
    print("="*40)

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(dates, cumulative_market, label=f'Market Benchmark ({total_return_market:.2%})', color='gray', alpha=0.5, linestyle='--')
    plt.plot(dates, cumulative_strategy, label=f'LSTM Strategy ({total_return_strategy:.2%})', color='red', linewidth=2)
    
    plt.title('Equity Curve: LSTM Strategy vs Market (2025)', fontsize=14, fontproperties=FONT_PROP)
    plt.ylabel('Normalized Value (Start=1)')
    plt.legend(loc='upper left', prop=FONT_PROP)
    plt.grid(True, alpha=0.3)
    
    # 标记买卖点 (可选，为了不让图太乱，只标这一行)
    # plt.scatter(dates[signals==1], cumulative_strategy[signals==1], marker='^', color='g', s=10, alpha=0.6)
    
    plt.gcf().autofmt_xdate()
    plt.savefig('Backtest_Result.png', dpi=300)
    print("✅ 资金曲线图已保存至: Backtest_Result.png") 
# ==============================================================================
# 5. 主程序逻辑
# ==============================================================================
def main():
    LOOK_BACK = 15 # 你之前用的15
    EPOCHS = 80
    BATCH_SIZE = 512
    N_ROUNDS = 5 

    # 使用你效果最好的结构
    EXPERIMENTS = {
        "Exp2_1": [64, 32],
    }

    # 1. 数据准备 (列数变了，Total features 要减去最后两列)
    df = get_and_prepare_data()
    TOTAL_FEATURES = df.shape[1] - 2 
    
    X_train_scaled, y_train_np, test_inputs_scaled, scaler, test_df_target = split_and_scale(df, LOOK_BACK)
    
    # 创建训练集
    X_train_np, y_train_np_window = create_xy(X_train_scaled, y_train_np, LOOK_BACK)
    
    # 创建测试集
    # [关键修改]: test_inputs_scaled 现在的列结构是 [特征..., Target, Price_Change]
    # 特征部分: [:, :-2]
    # 标签部分: [:, -2]
    # 收益率部分: [:, -1]
    
    X_test_np, y_test_np = create_xy(test_inputs_scaled[:, :-2], 
                                     test_inputs_scaled[:, -2].reshape(-1, 1), 
                                     LOOK_BACK)
    
    # 提取回测用的真实收益率 (对应 X_test 的时间段)
    # 因为 create_xy 会从 LOOK_BACK 开始截取，所以我们也从 LOOK_BACK 开始截取收益率
    test_returns_raw = test_inputs_scaled[LOOK_BACK:, -1]

    # 转 Tensor
    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_np_window, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32).to(device)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    real_labels_np = y_test_np.flatten()
    
    # 最佳模型保存逻辑
    best_acc = 0.0
    best_model_path = "best_model.pth"

    print(f"\n======== 开始 PyTorch 分类实验 ========")

    for exp_name, layers_config in EXPERIMENTS.items():
        print(f"\n>> [实验组]: {exp_name} 结构: {layers_config}")
        
        for i in range(N_ROUNDS):
            # ... (训练代码保持不变) ...
            set_seed(BASE_SEED + i)
            model = LSTMClassifier(input_size=TOTAL_FEATURES, hidden_layer_sizes=layers_config, output_size=1).to(device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

            for epoch in range(EPOCHS):
                model.train()
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    val_loss = criterion(model(X_test_tensor), y_test_tensor).item()
                scheduler.step(val_loss)

            # 评估
            model.eval()
            with torch.no_grad():
                pred_logits = model(X_test_tensor).cpu().numpy().flatten()
            
            acc = evaluate_predictions(real_labels_np, pred_logits)
            print(f"   Round {i+1}: Acc {acc:.4f}")
            
            # 保存最佳
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), best_model_path)

    # --- 🌟 核心回测环节 ---
    print("\n======== 开始回测 (Backtesting) ========")
    # 1. 重新加载最佳模型
    best_model = LSTMClassifier(input_size=TOTAL_FEATURES, hidden_layer_sizes=EXPERIMENTS["Exp2_1"], output_size=1).to(device)
    best_model.load_state_dict(torch.load(best_model_path))
    
    # 2. 运行回测
    # dates 需要对齐 (从 LOOK_BACK 开始)
    dates = test_df_target.index
    run_backtest(best_model, X_test_tensor, test_returns_raw, dates)

if __name__ == "__main__":
    test_df_raw = [] 
    main()   