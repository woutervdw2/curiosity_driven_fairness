# #Add parent folder
import sys
sys.path.append('../ml-fairness-gym')

#Lending environment without max bank cash
# from environments import lending
#Lending environment with max bank cash
from environments import lending_max3x as lending

from environments import lending_params

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from rl_agent import run_all

import itertools
import csv
import os


"""File that utilizes the parm testing file to train the final agents."""

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

LEARNING_STEPS = 100000



parm_csv = pd.read_csv('parms_test/results.csv')
parm_csv.columns = ['index', 'learning_rate', 'gamma', 'clip_range', 'n_steps', 'model', 'bank_cash']

#Choose best parameters
best_parm = parm_csv.loc[parm_csv['bank_cash'].idxmax()]
print(f"Best param config: {best_parm}")
best_parm = best_parm.drop('index')
best_parm = best_parm.drop('bank_cash')
best_parm = best_parm.drop('model')
best_parm = best_parm.to_dict()


#Train final agents
for model in tqdm(MODELS_TO_TRAIN):
    best_parm['rewards'] = model
    run_all(env_params, learning_steps=LEARNING_STEPS, show_plot=False, **best_parm)