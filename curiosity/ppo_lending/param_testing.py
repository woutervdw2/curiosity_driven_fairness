# #Add parent folder
import sys
ssys.path.append("/home/woutervdw2/Documents/thesis/code/ml-fairness-gym")

#Lending environment without max bank cash
# from environments import lending
#Lending environment with max bank cash
from environments import lending_max3x as lending

from environments import lending_params

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from rl_agent import run_all

import itertools
import csv
import os
"""File to run a parameter optimization on the lending environment."""

#Define environment
group_0_prob = 0.5
bank_starting_cash = np.float32(1000)
interest_rate = 1.0
cluster_shift_increment = 0.01
cluster_probabilities = lending_params.DELAYED_IMPACT_CLUSTER_PROBS

env_params = lending_params.DelayedImpactParams(
        applicant_distribution=lending_params.two_group_credit_clusters(
            cluster_probabilities= cluster_probabilities,
            group_likelihoods=[group_0_prob, 1 - group_0_prob]),
        bank_starting_cash=bank_starting_cash,
        interest_rate=interest_rate,
        cluster_shift_increment=cluster_shift_increment,
    )


MODELS_TO_TRAIN = ['visit_count', 'scalar', 'UCB']
LEARNING_STEPS = 10000

if not os.path.exists('parms_test/'):
    os.makedirs('parms_test/')

PATH = 'parms_test/'

#Define parameters to optimize
lr = [0.1, 0.01, 0.001, 0.0001]
gamma = [0.99, 0.9, 0.8, 0.7]
clip_range = [0.2, 0.1, 0.05, 0.01]
n_steps = [32, 64, 128, 256, 512]
model = 'UCB'

#Define parameter combinations
param_combinations = list(itertools.product(lr, gamma, clip_range, n_steps))

#Define csv file to save results
filename = PATH + 'results.csv'
fieldnames = ['index', 'lr', 'gamma', 'clip_range', 'n_steps', 'model', 'bank_cash']

#Check if file already exists
if os.path.exists(filename):
    with open(filename, mode='r') as f:
        reader = csv.DictReader(f, fieldnames=fieldnames)
        existing_combinations = set((float(row['lr']), float(row['gamma']), float(row['clip_range']),
                                      int(row['n_steps']), row['model']) for row in reader)
else:
    # If the file doesn't exist, create an empty set of existing parameter combinations
    existing_combinations = set()


#Run parameter optimization save results to csv file 
with open(filename, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not os.path.isfile(filename):
        print('File does not exist, creating new file')
        writer.writeheader()

    for i, params in enumerate(tqdm(param_combinations)):
        lr, gamma, clip_range, n_steps = params

        if (lr, gamma, clip_range, n_steps, model) in existing_combinations:
            continue

        test_results, baseline_results, action_callback = run_all(env_params, learning_steps=LEARNING_STEPS, 
                                                                    rewards=model, show_plot=False, verbose=0,
                                                                    learning_rate=lr,
                                                                    gamma=gamma,
                                                                    clip_range=clip_range,
                                                                    n_steps=n_steps,
                                                                    n_test_steps=1000)
        result = np.mean(test_results['bank_cash'])
        writer.writerow({'index': i, 'lr': lr, 'gamma': gamma, 'clip_range': clip_range, 
                            'n_steps': n_steps, 'model': model, 'bank_cash': result})
        f.flush()

