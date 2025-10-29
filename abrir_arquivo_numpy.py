import numpy as np

# Carregar o arquivo
matriz_transicao = np.load('transition_matrix_blackjack.npy')

# Informações sobre o array
print("Matriz de Transição:")
print(matriz_transicao)
print(f"Dimensões: {matriz_transicao.ndim}")
print(f"Formato: {matriz_transicao.shape}")
print(f"Tipo de dados: {matriz_transicao.dtype}")

matriz_recompensa = np.load('reward_matrix_blackjack.npy')
print("\nMatriz de Recompensas:")
print(matriz_recompensa)
print(f"Dimensões: {matriz_recompensa.ndim}")
print(f"Formato: {matriz_recompensa.shape}")
print(f"Tipo de dados: {matriz_recompensa.dtype}")