import numpy as np

n_states = 32 * 11 * 2
n_actions = 2
P = np.zeros((n_states, n_actions, n_states))
R = np.zeros((n_states, n_actions))
print(f'P shape: {P.shape}')
print(f'R shape: {R.shape}')
print(f'Initial P matrix sample:\n{P[0,0,:5]}')
print(f'Initial R matrix sample:\n{R[0,:]}')