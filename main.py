import gymnasium as gym
env = gym.make('Blackjack-v1', render_mode='human')
observation, info = env.reset()
done = False
while not done:
    env.render()
    observation, reward, done, truncated, info = env.step(action)
    state = observation
    player_sum,dealer_card,usable_ace = state
    if player_sum >= 17:
        action = 0
    else:
        action = 1
print(f'Reward final: {reward}')
print(done)
print(truncated)
print(info)
print(observation)
env.close()