# Código para estimar a matriz de transição de estados e recompensas no ambiente Blackjack usando o método de Monte Carlo
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
# Constantes e definições do ambiente
env = gym.make('Blackjack-v1', render_mode='rgb_array')
# 1 jogo = 1 episódio
# Método de monte carlo para estimar a matriz de transição de estados e recompensas
def estimate_transition_and_reward(env, num_episodes=100000):
    n_states = 32 * 11 * 2
    n_actions = 2
    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions))
    counts = np.zeros((n_states, n_actions))
    for episode in range(num_episodes):
        print(f'Iniciando o jogo {episode}')
        i = 0
        observation, info = env.reset()
        done = False
        while not done:
            env.render()
            state = observation
            player_sum,dealer_card,usable_ace = state
            print(f'Estado {i} -> Player sum: {player_sum}, Dealer card: {dealer_card}, Usable ace: {usable_ace}')
            action = env.action_space.sample()  # Ação aleatória
            print(f'Ação tomada: {"Stick" if action == 0 else "Hit"}')
            i += 1
            observation, reward, done, truncated, info = env.step(action)
            next_state = observation
            P[state, action, next_state] += 1
            R[state, action] += reward
            counts[state, action] += 1
    for s in range(n_states):
        for a in range(n_actions):
            if counts[s, a] > 0:
                P[s, a, :] /= counts[s, a]
                R[s, a] /= counts[s, a]
    return P, R

# plotar gráficos da matriz de transição e recompensas
P_dez, R_dez = estimate_transition_and_reward(env, num_episodes=10)
P_cem, R_cem = estimate_transition_and_reward(env, num_episodes=100)
P_mil, R_mil = estimate_transition_and_reward(env, num_episodes=1000)

print("Matriz de Transição após 10 episódios:")
print(P_dez)
print("Matriz de Recompensas após 100 episódios:")
print(R_cem)
print("Matriz de Transição após 1000 episódios:")
print(P_mil)
env.close()
