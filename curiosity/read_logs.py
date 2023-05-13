import numpy as np

b = np.load('models/ppo_lending/100_1UCB/UCB_logs/evaluations.npz', allow_pickle=True)
print(b.files)
print(b['results'])
print(b['ep_lengths'])