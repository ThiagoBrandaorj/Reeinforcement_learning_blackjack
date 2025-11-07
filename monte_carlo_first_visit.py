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
    return 3  # sempre fica em 0.5 do portfolio (posição 3 na lista [-1, 0, 0.25, 0.5, 0.75, 1])

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
num_episodes = 100

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

env.add_metric('Position Changes', lambda history : np.sum(np.diff(history['position']) != 0) )
env.add_metric("Episode Length", lambda history: len(history["position"]))
x = env.action_space
y = env.observation_space
print(f"Action space: {x}, Observation space: {y}")

returns_state = defaultdict(list) 
returns_sa = defaultdict(list)
V_medio_por_episodio = []
Q_medio_por_episodio = []
for ep in range(num_episodes):
    print(f"Iniciando episódio: {ep+1}")
    trajectory = []
    obs,info = env.reset()
    done = truncate = False
    # Coletar trajetória
    while not done and not truncate:
        disc_state = discretize(obs)
        action = politica_fixa(obs)
        next_obs, reward, done, truncate, info = env.step(action)
        trajectory.append((disc_state, reward, action))
        obs = next_obs
    # Monte carlo first visit
    visted_states = set()
    G = 0
    
    # Percorre do fim para o início
    for t in reversed(range(len(trajectory))):
        state,reward,action = trajectory[t]
        G = reward + GAMMA * G
        
        # First Visit -> só considera a 1ª ocorrência do estado
        if state not in visted_states:
            visted_states.add(state)
            if state not in returns_state:
                returns_state[state] = []
            returns_state[state].append(G)
            
            # média dos retornos
            V[state] = np.mean(returns_state[state])
            
            # méia dos retornos para Q(s,a)
            returns_sa[(state, action)].append(G)
            Q[(state, action)] = np.mean(returns_sa[(state, action)])
            
    # calcular média de V(s) após este episódio
    media_V = np.mean(list(V.values())) if len(V) > 0 else 0
    V_medio_por_episodio.append(media_V)
    # média de Q(s,a) após o episódio
    media_Q = np.mean(list(Q.values())) if len(Q) > 0 else 0
    Q_medio_por_episodio.append(media_Q)

end_time = time.time()
print(f"Tempo de execução: {end_time - start_time} segundos")
# resultados
print("Valores estimados de V(s):")
for s,valor in list(V.items())[:10]:  # apenas os primeiros 10 estados
    print(f"Estado: {s}, V(s): {valor}")
print("Valores estimados de Q(s,a):")
for (s,a),valor in list(Q.items())[:10]:  # apenas os primeiros 10 estados
    print(f"Estado: {s}, Ação: {a}, Q(s,a): {valor}")
    
# gráficos
states = list(V.keys())
values = list(V.values())
plt.figure(figsize=(12, 6))
plt.plot(V_medio_por_episodio)
plt.title("Convergência da Média de V(s) por Episódio")
plt.xlabel("Episódio")
plt.ylabel("V(s) Médio")
plt.show()
actions = list(Q.keys())
q_values = list(Q.values())
plt.figure(figsize=(12,6))
plt.plot(Q_medio_por_episodio)
plt.title("Convergência da Média de Q(s,a) por Episódio")
plt.xlabel("Episódio")
plt.ylabel("Q(s,a) Médio")
plt.grid(True)
plt.show()