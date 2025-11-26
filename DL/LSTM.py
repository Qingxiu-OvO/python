import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
# [设置]: 强制使用非交互式后端 Agg (保存图片专用)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
# [修改]: 引入 r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
import os
import random
from typing import List, Tuple

# ==============================================================================
# 0. 基础设置 (设备、随机种子 & 字体配置)
# ==============================================================================
# [关键]: 自动检测 Mac 的 MPS (Metal Performance Shaders) 加速
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
            print(f"✅ 字体配置成功: {FONT_NAME}")
            found_font = True
            break
    if not found_font:
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"❌ 字体配置异常: {e}")

plt.style.use('seaborn-v0_8') 

# ==============================================================================
# 1. 数据获取与特征工程
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
    
    # --- 技术指标 ---
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
    
    # --- 预测目标: 对数收益率 ---
    df['Log_Ret_Close'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Log_Ret_High'] = np.log(df['High'] / df['High'].shift(1))

    df = df.dropna()
    
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                    'MA10', 'MA20', 'RSI', 'MACD', 'Signal', 'BB_Upper', 'BB_Lower',
                    'Log_Ret_High', 'Log_Ret_Close']
    df = df[feature_cols]
    
    print(f"✅ 数据准备完成。特征数: {df.shape[1]-2}, 目标数: 2 (收益率)")
    return df

# ==============================================================================
# 2. 数据集处理
# ==============================================================================
def split_and_scale(df: pd.DataFrame, look_back: int) -> tuple[np.ndarray, np.ndarray, StandardScaler, pd.DataFrame]:
    train_df = df.loc['2020-01-01':'2024-12-31']
    test_df_raw = df.loc['2025-01-01':]

    if len(test_df_raw) == 0: sys.exit(1)

    full_dataset = pd.concat((train_df, test_df_raw), axis=0)
    test_inputs = full_dataset[len(full_dataset) - len(test_df_raw) - look_back:].values
    
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df.values)
    test_inputs_scaled = scaler.transform(test_inputs)
    
    return train_scaled, test_inputs_scaled, scaler, test_df_raw

def create_xy(dataset: np.ndarray, look_back: int) -> tuple[np.ndarray, np.ndarray]:
    X, Y = [], []
    idx_high_ret = -2
    idx_close_ret = -1
    
    for i in range(look_back, len(dataset)):
        X.append(dataset[i-look_back:i, :])
        Y.append([dataset[i, idx_high_ret], dataset[i, idx_close_ret]]) 
        
    return np.array(X), np.array(Y)

# ==============================================================================
# 3. 模型构建 (PyTorch 版)
# ==============================================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size=2, dropout_prob=0.3):
        super(LSTMModel, self).__init__()
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # 动态构建多层 LSTM
        prev_size = input_size
        for hidden_size in hidden_layer_sizes:
            # batch_first=True 使得输入形状为 (batch, seq, feature)
            self.layers.append(nn.LSTM(prev_size, hidden_size, batch_first=True))
            self.dropouts.append(nn.Dropout(dropout_prob))
            prev_size = hidden_size
            
        # 最终输出层
        self.fc = nn.Linear(prev_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out = x
        for i in range(len(self.layers)):
            out, _ = self.layers[i](out) # out shape: (batch, seq, hidden)
            out = self.dropouts[i](out)
            
        # 取最后一个时间步的输出
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

# ==============================================================================
# 4. 价格还原逻辑
# ==============================================================================
def recover_prices(pred_returns_scaled: np.ndarray, scaler: StandardScaler, 
                   prev_prices_high: np.ndarray, prev_prices_close: np.ndarray, 
                   feature_total_count: int) -> tuple[np.ndarray, np.ndarray]:
    
    dummy = np.zeros((len(pred_returns_scaled), feature_total_count))
    dummy[:, -2] = pred_returns_scaled[:, 0]
    dummy[:, -1] = pred_returns_scaled[:, 1]
    
    res_unscaled = scaler.inverse_transform(dummy)
    pred_log_ret_high = res_unscaled[:, -2]
    pred_log_ret_close = res_unscaled[:, -1]
    
    rec_high = prev_prices_high * np.exp(pred_log_ret_high)
    rec_close = prev_prices_close * np.exp(pred_log_ret_close)
    
    return rec_high, rec_close

# [修改]: 增加返回 R2 Score
def evaluate_predictions(real: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    rmse = np.sqrt(mean_squared_error(real, pred))
    mae = mean_absolute_error(real, pred)
    r2 = r2_score(real, pred) # 计算 R平方
    return rmse, mae, r2

# ==============================================================================
# 5. 主程序逻辑
# ==============================================================================
def main():
    LOOK_BACK = 30
    EPOCHS = 80
    BATCH_SIZE = 512
    N_ROUNDS = 5 

    EXPERIMENTS = {
        "Exp2_1": [256, 128],
        "Exp2_2": [512, 256],
        "Exp3_1": [256, 128, 64],
        "Exp3_2": [512, 256, 128],
    }

    # 1. 数据准备
    df = get_and_prepare_data()
    TOTAL_FEATURES = df.shape[1]
    
    train_scaled, test_inputs_scaled, scaler, test_df_target = split_and_scale(df, LOOK_BACK)
    
    # 转换为 Numpy
    X_train_np, y_train_np = create_xy(train_scaled, LOOK_BACK)
    X_test_np, y_test_np = create_xy(test_inputs_scaled, LOOK_BACK)

    # 转换为 PyTorch Tensor 并移动到设备 (MPS)
    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32).to(device)

    # 使用 DataLoader 构建批次
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 还原基准数据
    full_target_prices = pd.concat([
        df.loc['2024-12-01':].iloc[-(len(test_df_target)+1):],
    ])
    prev_date = df.index[df.index.get_loc(test_df_target.index[0]) - 1]
    
    ref_prices_high = df['High'].loc[prev_date : test_df_target.index[-2]].values
    ref_prices_close = df['Close'].loc[prev_date : test_df_target.index[-2]].values
    
    real_close = test_df_target['Close'].values
    real_high = test_df_target['High'].values
    dates = test_df_target.index

    final_results_summary = []

    print(f"\n======== 开始 PyTorch 实验 (含 R2 得分) ========")

    for exp_name, layers_config in EXPERIMENTS.items():
        print(f"\n>> [实验组]: {exp_name} 结构: {layers_config}")
        
        temp_maes = []
        temp_rmses = []
        temp_r2s = [] # [修改]: 记录 R2
        temp_pred_high_list = []
        temp_pred_close_list = []

        for i in range(N_ROUNDS):
            print(f"   - 第 {i+1}/{N_ROUNDS} 次训练...", end="", flush=True)
            
            set_seed(BASE_SEED + i)
            
            model = LSTMModel(input_size=X_train_np.shape[2], hidden_layer_sizes=layers_config).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

            # --- 训练循环 ---
            for epoch in range(EPOCHS):
                model.train()
                train_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_test_tensor)
                    val_loss = criterion(val_outputs, y_test_tensor).item()
                
                scheduler.step(val_loss)

            # --- 预测 ---
            model.eval()
            with torch.no_grad():
                pred_rets_scaled_tensor = model(X_test_tensor)
                pred_rets_scaled = pred_rets_scaled_tensor.cpu().numpy()
            
            # 2. 还原
            rec_high, rec_close = recover_prices(
                pred_rets_scaled, scaler, 
                ref_prices_high, ref_prices_close, 
                TOTAL_FEATURES
            )
            
            # 3. 评估 [修改]: 接收 r2
            rmse, mae, r2 = evaluate_predictions(real_close, rec_close)
            
            temp_maes.append(mae)
            temp_rmses.append(rmse)
            temp_r2s.append(r2)
            temp_pred_high_list.append(rec_high)
            temp_pred_close_list.append(rec_close)
            
            print(f" 完成. (MAE: {mae:.2f}, R2: {r2:.4f})")

        # 计算平均
        avg_mae = np.mean(temp_maes)
        avg_rmse = np.mean(temp_rmses)
        avg_r2 = np.mean(temp_r2s) # [修改]: 计算平均 R2
        avg_pred_high = np.mean(np.array(temp_pred_high_list), axis=0)
        avg_pred_close = np.mean(np.array(temp_pred_close_list), axis=0)
        
        print(f"   >> {exp_name} 平均 MAE: {avg_mae:.4f}, 平均 R2: {avg_r2:.4f}")
        
        final_results_summary.append({
            "Experiment": exp_name,
            "Structure": str(layers_config),
            "Avg_MAE": avg_mae,
            "Avg_RMSE": avg_rmse,
            "Avg_R2": avg_r2, # [修改]: 添加到结果摘要
            "Pred_High": avg_pred_high,
            "Pred_Close": avg_pred_close
        })

    results_df = pd.DataFrame(final_results_summary).sort_values(by="Avg_MAE")
    
    print("\n" + "="*80)
    print(f"最终实验报告 (PyTorch MPS 版)")
    print("="*80)
    # [修改]: 打印包含 R2 的表格
    print(results_df[["Experiment", "Structure", "Avg_MAE", "Avg_RMSE", "Avg_R2"]].to_string(index=False))
    
    best_exp = results_df.iloc[0]
    best_name = best_exp["Experiment"]
    print(f"\n🏆 最佳模型方案: {best_name} (平均MAE: {best_exp['Avg_MAE']:.4f}, R2: {best_exp['Avg_R2']:.4f})")

    # 绘图
    print(f"\n正在绘制最佳模型 ({best_name}) 的结果...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    ax1.set_title(f'上证 2025 最高价预测 (基于收益率预测还原, {best_name})', fontsize=14, fontproperties=FONT_PROP)
    ax1.plot(dates, real_high, label='实际最高价', color='#d62728', linewidth=2)
    ax1.plot(dates, best_exp["Pred_High"], label='预测最高价', color='#1f77b4', linestyle='--', linewidth=1.5)
    ax1.legend(loc='upper left', prop=FONT_PROP)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel('价格', fontproperties=FONT_PROP)

    ax2.set_title(f'上证 2025 收盘价预测 (基于收益率预测还原, {best_name})', fontsize=14, fontproperties=FONT_PROP)
    ax2.plot(dates, real_close, label='实际收盘价', color='#2ca02c', linewidth=2)
    ax2.plot(dates, best_exp["Pred_Close"], label='预测收盘价', color='#ff7f0e', linestyle='--', linewidth=1.5)
    ax2.legend(loc='upper left', prop=FONT_PROP)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel('价格', fontproperties=FONT_PROP)
    ax2.set_xlabel('日期', fontproperties=FONT_PROP)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    
    SAVE_NAME = f'PyTorch_Result_{best_name}.png'
    plt.savefig(SAVE_NAME, dpi=300)
    print(f"✅ 图表已保存至: {SAVE_NAME}")

if __name__ == "__main__":
    test_df_raw = [] 
    main()