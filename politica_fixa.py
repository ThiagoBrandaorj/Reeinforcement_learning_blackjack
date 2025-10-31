import gymnasium as gym
import numpy as np
import gym_trading_env
from gym_trading_env.downloader import download
import datetime
import pandas as pd
import os
import time


# guarda o dataframe em um arquivo pickle para uso posterior
df = pd.read_pickle("./data/binance-BTCUSDT-1h.pkl")


# cria a coluna feature_close: (close[t] - close[t-1]) / close[t-1]
df["feature_close"] = df["close"].pct_change()

# cria a coluna feature_open: open[t] / close[t]
df["feature_open"] = df["open"] / df["close"]

# cria a coluna feature_high: high[t] / close[t]
df["feature_high"] = df["high"] / df["close"]

# cria a coluna feature_low: low[t] / close[t]
df["feature_low"] = df["low"] / df["close"]

# Create the feature: volume[t] / max(*volume[t-7*24:t+1])
# Try common volume column names - adjust based on the actual column names
if 'volume' in df.columns:
    df["feature_volume"] = df["volume"] / df["volume"].rolling(7*24).max()
elif 'Volume' in df.columns:
    df["feature_volume"] = df["Volume"] / df["Volume"].rolling(7*24).max()
elif 'Volume USD' in df.columns:
    df["feature_volume"] = df["Volume USD"] / df["Volume USD"].rolling(7*24).max()
else:
    # If no volume column is found, use a placeholder or skip this feature
    print("Warning: No volume column found. Using close price as volume proxy.")
    df["feature_volume"] = df["close"] / df["close"].rolling(7*24).max()

df.dropna(inplace=True)  # Clean again!

print(f"Final DataFrame shape: {df.shape}")
print("Available features:", [col for col in df.columns if col.startswith('feature_')])
with pd.option_context('display.max_columns', None, 'display.width', None):
    print(df.head(10))
# Each step, the environment will return 5 inputs: "feature_close", "feature_open", "feature_high", "feature_low", "feature_volume"