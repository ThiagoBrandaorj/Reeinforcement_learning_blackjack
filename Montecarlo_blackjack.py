# Código para estimar a matriz de transição de estados e recompensas no ambiente Blackjack usando o método de Monte Carlo
import gymnasium as gym
import numpy as np
import time
# Constantes e definições do ambiente
start_time = time.time()
env = gym.make('Blackjack-v1', render_mode='human')
# 1 jogo = 1 episódio
# Método de monte carlo para estimar a matriz de transição de estados e recompensas
def estimate_transition_and_reward(env, num_episodes=10000):
    n_states = 32 * 11 * 2
    n_actions = 2
    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions))
    counts = np.zeros((n_states, n_actions))
    for episode in range(num_episodes):
        print(f'Iniciando o jogo {episode}')
        i = 0
        done = False
        observation, info = env.reset()
        while not done:
            env.render()
            state = observation
            player_sum,dealer_card,usable_ace = state
            print(f'Estado {i} do jogo {episode} -> Player sum: {player_sum}, Dealer card: {dealer_card}, Usable ace: {usable_ace}')
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
P,R = estimate_transition_and_reward(env)
print("Matriz de Transição após 10.000 episódios:")
print(P)
# salvar a matriz de transição em um arquivo .npy
np.save('transition_matrix_blackjack.npy', P)
print("Matriz de Recompensas após 10.000 episódios:")
print(R)
# salvar a matriz de recompensas em um arquivo .npy
np.save('reward_matrix_blackjack.npy', R)
env.close()
end_time = time.time()
elapsed_time = end_time - start_time
print(f'Tempo total de execução: {elapsed_time:.2f} segundos')
