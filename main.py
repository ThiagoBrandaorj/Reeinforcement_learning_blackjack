import gymnasium as gym
env = gym.make('Blackjack-v1', render_mode='human')
results = []
num_games = 1000
for game in range(num_games):
    i = 0
    done = False
    print(f'Iniciando o jogo {game}')
    observation, info = env.reset()
    while not done:
        env.render()
        state = observation
        player_sum,dealer_card,usable_ace = state
        print(f'Estado {i} -> Player sum: {player_sum}, Dealer card: {dealer_card}, Usable ace: {usable_ace}')
        if player_sum >= 17:
            action = 0
        else:
            action = 1
        print(f'Ação tomada: {"Stick" if action == 0 else "Hit"}')
        i += 1
        observation, reward, done, truncated, info = env.step(action)
        if done:
            results.append(reward)
        else:
            continue
        print(f'Recompa final do episódio: {reward}')
        print(f'observation final do episódio: {observation}')
env.close()
print(len(results))