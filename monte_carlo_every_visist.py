import gymnasium as gym
import numpy as np
import gym_trading_env
from gym_trading_env.downloader import download
import random
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import os
import time
from collections import defaultdict

# Começo da medição do tempo de execução do código
start_time = time.time()
GAMMA = 0.99  # fator de desconto do cálculo do retorno

# Função de recompensa personalizada: log retorno do portfólio
def reward_function(history):
    return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])

def politica_fixa(observation):
    if random.random() < 0.3:
        return random.randint(0,5)  # ação aleatória 30% do tempo
    return 3  # sempre fica em 0.5 do portfolio

# Função para discretizar o estado
def discretize(obs):
    fc, fo, fh, fl, fv, cp, lp = obs
    return (
        round(float(fc), 2),
        round(float(fo), 2),
        round(float(fh), 2),
        round(float(fl), 2),
        round(float(fv), 2),
        int(cp),
        int(lp)
    )

# Valores de V(s)
V = {}

# Valores de Q(s,a)
Q = {}

# Número de episodios
num_episodes = 50

# cria uma pasta data se não existir
os.makedirs("data", exist_ok=True)

# Download de dados históricos do BTC/USDT
download(exchange_names=["binance"],
    symbols=["BTC/USDT"],
    timeframe="1h",
    dir="data",
    since=datetime.datetime(year=2020, month=1, day=1),
)

# guarda o dataframe
df = pd.read_pickle("./data/binance-BTCUSDT-1h.pkl")

# cria features
df["feature_close"] = df["close"].pct_change()
df["feature_open"] = df["open"] / df["close"]
df["feature_high"] = df["high"] / df["close"]
df["feature_low"] = df["low"] / df["close"]

if 'volume' in df.columns:
    df["feature_volume"] = df["volume"] / df["volume"].rolling(7*24).max()
else:
    df["feature_volume"] = df["close"] / df["close"].rolling(7*24).max()

df.dropna(inplace=True)

# Criar ambiente
env = gym.make(
    "TradingEnv",
    name="BTCUSD",
    df=df,
    positions=[-1, 0, 0.25, 0.5, 0.75, 1],
    trading_fees=0.001/100,
    borrow_interest_rate=0.0003/100,
    reward_function=reward_function,
)

env.add_metric('Position Changes', lambda history : np.sum(np.diff(history['position']) != 0) )
env.add_metric("Episode Length", lambda history: len(history["position"]))

returns_state = defaultdict(list)
returns_sa = defaultdict(list)

# ======================================================================
#             ✅ MONTE CARLO EVERY-VISIT IMPLEMENTADO AQUI
# ======================================================================

for ep in range(num_episodes):
    print(f"Iniciando episódio: {ep+1}")
    
    trajectory = []
    obs,info = env.reset()
    done = truncate = False

    # Coleta da trajetória completa
    while not done and not truncate:
        state = discretize(obs)
        action = politica_fixa(obs)
        next_obs, reward, done, truncate, info = env.step(action)
        trajectory.append((state, reward, action))
        obs = next_obs

    # Calcular retornos (G)
    G = 0
    for t in reversed(range(len(trajectory))):
        s, r, a = trajectory[t]
        G = r + GAMMA * G

        # Every-visit → sempre adiciona
        returns_state[s].append(G)
        V[s] = np.mean(returns_state[s])

        returns_sa[(s,a)].append(G)
        Q[(s,a)] = np.mean(returns_sa[(s,a)])

end_time = time.time()
print(f"Tempo de execução: {end_time - start_time} segundos")

# resultados
print("Valores estimados de V(s):")
for s,valor in list(V.items())[:10]:
    print(f"Estado: {s}, V(s): {valor}")

print("Valores estimados de Q(s,a):")
for (s,a),valor in list(Q.items())[:10]:
    print(f"Estado: {s}, Ação: {a}, Q(s,a): {valor}")

# gráficos
states = list(V.keys())
values = list(V.values())

plt.figure(figsize=(12, 6))
plt.scatter(range(len(states)), values, alpha=0.6)
plt.title("Estimated State-Value Function V(s) — Every Visit")
plt.xlabel("States")
plt.ylabel("V(s)")
plt.show()
actions = list(Q.keys())
q_values = list(Q.values())
plt.figure(figsize=(12, 6))
plt.scatter(range(len(actions)), q_values, alpha=0.6)
plt.title("Estimated Action-Value Function Q(s,a) — Every Visit")
plt.xlabel("State-Action Pairs")
plt.ylabel("Q(s,a)")
plt.show()
