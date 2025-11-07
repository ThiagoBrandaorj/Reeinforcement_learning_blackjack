import os
import time
import datetime
import numpy as np
import pandas as pd
import gymnasium as gym
import gym_trading_env
from gym_trading_env.downloader import download
from collections import defaultdict

# Configurações
GAMMA = 0.95  # Fator de desconto para valorizar recompensas futuras
EPISODES = 700
MAX_STEPS = 200
ROUND_NDIGITS = 2  # Discretização
EPSILON_START = 0.3  # Exploração inicial alta
EPSILON_MIN = 0.05  # Exploração mínima
EPSILON_DECAY = 0.995  # Decaimento gradual
SEED = 42
THETA = 0.001  # Convergência mais rigorosa
INITIAL_PORTFOLIO = 5000

np.random.seed(SEED)
start_time = time.time()

# Preparação dos Dados
os.makedirs("data", exist_ok=True)
download(
    exchange_names=["binance"],
    symbols=["BTC/USDT"],
    timeframe="1h",
    dir="data",
    since=datetime.datetime(2020, 1, 1),
)

df = pd.read_pickle("./data/binance-BTCUSDT-1h.pkl")
print(f"Dataset carregado: {df.shape[0]} linhas")

df = df.iloc[:5000]

# Features técnicas
df["feature_close"] = df["close"].pct_change()
df["feature_open"] = df["open"] / df["close"]
df["feature_high"] = df["high"] / df["close"]
df["feature_low"] = df["low"] / df["close"]

# Volume normalizado
if "volume" in df.columns:
    vol = df["volume"]
elif "Volume" in df.columns:
    vol = df["Volume"]
else:
    vol = df["close"]

df["feature_volume"] = vol / vol.rolling(24, min_periods=1).max()

# taxa de mudança
df["feature_momentum"] = df["close"].pct_change(periods=5)

df.dropna(inplace=True)
print(f"Features criadas: {df.shape}")

# Ambiente
env = gym.make(
    "TradingEnv",
    df=df,
    positions=[-1, 0, 1], # Short, Neutral, Long
    trading_fees=0.01,
    borrow_interest_rate=0.0003,
    portfolio_initial_value=INITIAL_PORTFOLIO,
)

# Funções Auxiliares
def get_portfolio_value(info):
    """Extrai o valor do portfólio do info do ambiente"""
    if 'portfolio_valuation' in info:
        return info['portfolio_valuation']
    elif hasattr(env.unwrapped, 'historical_info'):
        history = env.unwrapped.historical_info
        if history and 'portfolio_valuation' in history:
            return history['portfolio_valuation'][-1]
    return INITIAL_PORTFOLIO

def discretize_state(obs, portfolio_value, ndigits=ROUND_NDIGITS):
    """
    Discretiza a observação incluindo métricas de performance
    """
    if isinstance(obs, dict):
        features = []
        for key in sorted(obs.keys()):
            if 'feature' in key:
                features.append(obs[key])
        obs_array = np.array(features)
    else:
        obs_array = np.asarray(obs).ravel().astype(float)
    
    # Normaliza o valor do portfólio (% de retorno)
    portfolio_return = (portfolio_value - INITIAL_PORTFOLIO) / INITIAL_PORTFOLIO
    
    # Categoria de performance
    if portfolio_return > 0.05:
        perf_category = 2  # Excelente
    elif portfolio_return > 0:
        perf_category = 1  # Positivo
    elif portfolio_return > -0.05:
        perf_category = 0  # Neutro
    else:
        perf_category = -1  # Negativo
    
    # Combina features com performance
    state_array = np.append(obs_array, [portfolio_return, perf_category])
    
    return tuple(np.round(state_array, decimals=ndigits).tolist())

def epsilon_greedy_policy(Q, state, epsilon):
    """Política epsilon-greedy otimizada"""
    if np.random.random() < epsilon:
        return env.action_space.sample()
    
    # Inicializa ações não vistas com valores otimistas
    if state not in Q or len(Q[state]) == 0:
        for action in range(env.action_space.n):
            Q[state][action] = 0.1 # Valor otimista inicial
    
    # Retorna ação com maior Q-value
    return max(Q[state], key=Q[state].get)

# 1. POLICY ITERATION
print("\n" + "="*50)
print("Policy Iteration")
print("="*50)

V_pi = defaultdict(float)
policy_pi = defaultdict(lambda: 1)  # Política inicial: neutral (ação 1)
state_transitions = defaultdict(lambda: defaultdict(list))  # Armazena transições

# Coletar estados e transições do ambiente
print("Coletando estados e transições...")
states_visited = set()

for episode in range(20):
    obs, info = env.reset(seed=SEED + episode)
    portfolio_value = get_portfolio_value(info)
    state = discretize_state(obs, portfolio_value)
    states_visited.add(state)
    
    for step in range(MAX_STEPS):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        next_portfolio = get_portfolio_value(next_info)
        next_state = discretize_state(next_obs, next_portfolio)
        
        # Armazena transição: (reward, next_state)
        state_transitions[state][action].append((reward, next_state))
        
        states_visited.add(next_state)
        state = next_state
        
        if terminated or truncated:
            break

print(f"Estados únicos coletados: {len(states_visited)}")
print(f"Transições registradas: {sum(len(state_transitions[s]) for s in state_transitions)}")

# Policy Iteration
iteration = 0
max_iterations = 20

while iteration < max_iterations:
    print(f"\n--- Iteração {iteration + 1} ---")
    
    # AVALIAÇÃO DA POLÍTICA
    delta = float('inf')
    eval_iter = 0
    
    while delta > THETA and eval_iter < 200:
        delta = 0
        
        for state in states_visited:
            if state not in state_transitions:
                continue
                
            v = V_pi[state]
            action = policy_pi[state]
            
            # Calcula valor esperado baseado nas transições observadas
            if action in state_transitions[state] and len(state_transitions[state][action]) > 0:
                expected_value = 0
                for reward, next_state in state_transitions[state][action]:
                    expected_value += reward + GAMMA * V_pi[next_state]
                expected_value /= len(state_transitions[state][action])
                V_pi[state] = expected_value
            
            delta = max(delta, abs(v - V_pi[state]))
        
        eval_iter += 1
    
    print(f"   Avaliação: {eval_iter} iterações (delta={delta:.6f})")
    
    # MELHORIA DA POLÍTICA
    policy_stable = True
    
    for state in states_visited:
        if state not in state_transitions:
            continue
            
        old_action = policy_pi[state]
        
        # Calcula Q(s,a) para cada ação
        action_values = {}
        for action in range(env.action_space.n):
            if action in state_transitions[state] and len(state_transitions[state][action]) > 0:
                q_value = 0
                for reward, next_state in state_transitions[state][action]:
                    q_value += reward + GAMMA * V_pi[next_state]
                action_values[action] = q_value / len(state_transitions[state][action])
            else:
                action_values[action] = V_pi[state]  # Fallback
        
        # Escolhe ação com maior Q-value
        best_action = max(action_values, key=action_values.get)
        policy_pi[state] = best_action
        
        if old_action != best_action:
            policy_stable = False
    
    print(f"   Política {'estável' if policy_stable else 'atualizada'}")
    
    iteration += 1
    if policy_stable:
        print("Política convergiu!")
        break

print(f"\nPolicy Iteration: {iteration} iterações, {len(V_pi)} estados avaliados")

# 2. MONTE CARLO FIRST-VISIT
print("\nMonte Carlo First-Visit")
print("="*50)

Q_first = defaultdict(lambda: defaultdict(float))
returns_first = defaultdict(lambda: defaultdict(list))
epsilon = EPSILON_START

for episode in range(EPISODES):
    episode_data = []
    obs, info = env.reset(seed=SEED + episode)
    portfolio_value = get_portfolio_value(info)
    state = discretize_state(obs, portfolio_value)
    
    episode_reward = 0
    
    for step in range(MAX_STEPS):
        action = epsilon_greedy_policy(Q_first, state, epsilon)
        obs, reward, terminated, truncated, info = env.step(action)
        
        portfolio_value = get_portfolio_value(info)
        next_state = discretize_state(obs, portfolio_value)
        
        episode_data.append((state, action, reward))
        episode_reward += reward
        state = next_state
        
        if terminated or truncated:
            break
    
    # Calcular retornos (first-visit)
    G = 0
    visited = set()
    
    for state, action, reward in reversed(episode_data):
        G = reward + GAMMA * G
        
        if (state, action) not in visited:
            visited.add((state, action))
            returns_first[state][action].append(G)
            Q_first[state][action] = np.mean(returns_first[state][action])
    
    # Decaimento de epsilon
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    
    if (episode + 1) % 50 == 0:
        avg_q = np.mean([Q_first[s][a] for s in Q_first for a in Q_first[s]])
        print(f"Ep {episode + 1}/{EPISODES} | Estados: {len(Q_first)} | "
              f"Q médio: {avg_q:.4f} | ε: {epsilon:.3f} | Recompensa: {episode_reward:.2f}")

print(f"MC First-Visit: {len(Q_first)} estados aprendidos")

# 3. MONTE CARLO EVERY-VISIT
print("\nMonte Carlo Every-Visit")
print("="*50)

Q_every = defaultdict(lambda: defaultdict(float))
returns_every = defaultdict(lambda: defaultdict(list))
epsilon = EPSILON_START

for episode in range(EPISODES):
    episode_data = []
    obs, info = env.reset(seed=SEED + episode)
    portfolio_value = get_portfolio_value(info)
    state = discretize_state(obs, portfolio_value)
    
    episode_reward = 0
    
    for step in range(MAX_STEPS):
        action = epsilon_greedy_policy(Q_every, state, epsilon)
        obs, reward, terminated, truncated, info = env.step(action)
        
        portfolio_value = get_portfolio_value(info)
        next_state = discretize_state(obs, portfolio_value)
        
        episode_data.append((state, action, reward))
        episode_reward += reward
        state = next_state
        
        if terminated or truncated:
            break
    
    # Calcular retornos (every-visit)
    G = 0
    for state, action, reward in reversed(episode_data):
        G = reward + GAMMA * G
        returns_every[state][action].append(G)
        Q_every[state][action] = np.mean(returns_every[state][action])
    
    # Decaimento de epsilon
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    
    if (episode + 1) % 50 == 0:
        avg_q = np.mean([Q_every[s][a] for s in Q_every for a in Q_every[s]])
        print(f"Ep {episode + 1}/{EPISODES} | Estados: {len(Q_every)} | "
              f"Q médio: {avg_q:.4f} | ε: {epsilon:.3f} | Recompensa: {episode_reward:.2f}")

print(f"MC Every-Visit: {len(Q_every)} estados aprendidos")

# 4. TESTE DOS MODELOS
print("\nTeste dos Modelos")
print("="*50)

def test_policy(policy_dict, Q_dict, name, episodes=20):
    """Testa uma política aprendida"""
    total_rewards = []
    final_portfolios = []
    returns = []
    
    for ep in range(episodes):
        obs, info = env.reset(seed=SEED + 1000 + ep)
        portfolio_value = get_portfolio_value(info)
        episode_reward = 0
        
        for step in range(MAX_STEPS):
            state = discretize_state(obs, portfolio_value)
            
            # Escolhe ação (sem exploração no teste)
            if state in policy_dict and isinstance(policy_dict[state], int):
                action = policy_dict[state]
            elif state in Q_dict and len(Q_dict[state]) > 0:
                action = max(Q_dict[state], key=Q_dict[state].get)
            else:
                action = 1  # Neutral como padrão
            
            obs, reward, terminated, truncated, info = env.step(action)
            portfolio_value = get_portfolio_value(info)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
        final_portfolios.append(portfolio_value)
        returns.append((portfolio_value - INITIAL_PORTFOLIO) / INITIAL_PORTFOLIO * 100)
    
    avg_reward = np.mean(total_rewards)
    avg_portfolio = np.mean(final_portfolios)
    avg_return = np.mean(returns)
    
    print(f"\n{name}:")
    print(f"  Recompensa Média: {avg_reward:.4f} (±{np.std(total_rewards):.4f})")
    print(f"  Portfólio Final: ${avg_portfolio:.2f}")
    print(f"  Retorno Médio: {avg_return:+.2f}% (±{np.std(returns):.2f}%)")
    print(f"  Melhor resultado: ${max(final_portfolios):.2f}")
    print(f"  Pior resultado: ${min(final_portfolios):.2f}")
    
    return avg_reward, avg_portfolio, avg_return

# Testar cada modelo
pi_reward, pi_portfolio, pi_return = test_policy(policy_pi, {}, "Policy Iteration")
first_reward, first_portfolio, first_return = test_policy({}, Q_first, "MC First-Visit")
every_reward, every_portfolio, every_return = test_policy({}, Q_every, "MC Every-Visit")

# 5. ANÁLISE COMPARATIVA
print("\nAnálise Comparativa Final")
print("="*50)

results = {
    "Policy Iteration": {"reward": pi_reward, "portfolio": pi_portfolio, "return": pi_return},
    "MC First-Visit": {"reward": first_reward, "portfolio": first_portfolio, "return": first_return},
    "MC Every-Visit": {"reward": every_reward, "portfolio": every_portfolio, "return": every_return}
}

# Ranking por recompensa
print("\nRanking por Recompensa:")
sorted_by_reward = sorted(results.items(), key=lambda x: x[1]["reward"], reverse=True)
for i, (name, metrics) in enumerate(sorted_by_reward, 1):
    print(f"{i}. {name}: {metrics['reward']:.4f}")

# Ranking por retorno
print("\nRanking por Retorno (%):")
sorted_by_return = sorted(results.items(), key=lambda x: x[1]["return"], reverse=True)
for i, (name, metrics) in enumerate(sorted_by_return, 1):
    print(f"{i}. {name}: {metrics['return']:+.2f}%")

# Melhor modelo geral
best_model = max(results.items(), key=lambda x: x[1]["reward"])
print(f"\nMelhor Modelo: {best_model[0]}")
print(f"   Recompensa: {best_model[1]['reward']:.4f}")
print(f"   Retorno: {best_model[1]['return']:+.2f}%")

# Resultados Finais
print("\nResumo Final")
print(f"Tempo de execução: {time.time() - start_time:.2f}s")
print(f"Policy Iteration: {len(V_pi)} estados")
print(f"MC First-Visit: {len(Q_first)} estados")
print(f"MC Every-Visit: {len(Q_every)} estados")
