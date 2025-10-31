import gymnasium as gym
import numpy as np
import gym_trading_env
from gym_trading_env.downloader import download
import datetime
import pandas as pd
import os
import time
# Começo da medição do tempo de execução
start_time = time.time()

# Função de recompensa personalizada: log retorno do portfólio
def reward_function(history):
    return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])

# cria uma pasta data se não existir
os.makedirs("data", exist_ok=True)

# Download de dados históricos do BTC/USDT de 1 em 1 hora desde 01-01-2020
download(exchange_names=["binance"],
    symbols=["BTC/USDT"],
    timeframe="1h",
    dir="data",
    since=datetime.datetime(year=2020, month=1, day=1),
)

# guarda o dataframe em um arquivo pickle para uso posterior
df = pd.read_pickle("./data/binance-BTCUSDT-1h.pkl")

# nomes das colunas que vieram da binance
print("DataFrame columns:", df.columns.tolist())

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
print(df.head(10))
# Each step, the environment will return 5 inputs: "feature_close", "feature_open", "feature_high", "feature_low", "feature_volume"

# name - moneda para trading (ex: "BTCUSD", "ETHUSD", "AAPL", etc)
# df - dataframe com os dados históricos e features customizadas
# positions - lista de posições possíveis (ex: [-1, 0, 1] para SHORT, OUT, LONG)
# trading_fees - taxa de trading por compra/venda (ex: 0.01/100 para 0.01%)
# borrow_interest_rate - taxa de juros por timestep quando em posição SHORT (ex: 0.0003/100 para 0.0003%)
env = gym.make("TradingEnv",
        name="BTCUSD",
        df=df,  # Your dataset with your custom features
        positions=[-1, 0, 0.25, 0.5, 0.75, 1],  # -1 (=SHORT), 0(=OUT), +1 (=LONG)
        trading_fees=0.001/100,  # 0.001% per stock buy / sell (Binance fees)
        borrow_interest_rate=0.0003/100,  # 0.0003% per timestep (one timestep = 1h here)
        reward_function=reward_function,  # custom reward function defined above
    )
env.add_metric("Episode Length", lambda history: len(history["position"]))
x = env.action_space
y = env.observation_space
print(f"Action space: {x}, Observation space: {y}")
# Roda um episódio de trading até ele acabar
done, truncated = False, False
observation, info = env.reset()
while not done and not truncated:
    position_index = env.action_space.sample()  #  A cada passo de tempo pega um índice de posição aleatório da sua lista de posições (=[-1, 0, 0.25, 0.5, 0.75, 1])
    observation, reward, done, truncated, info = env.step(position_index)
end_time = time.time()
print(f"Episode finished in {end_time - start_time:.2f} seconds.")