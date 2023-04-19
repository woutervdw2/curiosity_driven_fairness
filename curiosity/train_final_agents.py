# #Add parent folder
import sys
sys.path.append('../ml-fairness-gym')

#Lending environment without max bank cash
# from environments import lending
#Lending environment with max bank cash
from environments import lending_max3x as lending

from environments import lending_params

from compare_agents import grouped_barplot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from rl_agent import run_all



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

def choose_parms():
    """Chooses the best parameters from the parm testing file."""
    parm_csv = pd.read_csv('parms_test/results.csv')
    parm_csv.columns = ['index', 'learning_rate', 'gamma', 'clip_range', 'n_steps', 'model', 'bank_cash']

    #Choose best parameters
    best_parm = parm_csv.loc[parm_csv['bank_cash'].idxmax()]
    print(f"Best param config: {best_parm}")
    best_parm = best_parm.drop('index')
    best_parm = best_parm.drop('bank_cash')
    best_parm = best_parm.drop('model')
    best_parm = best_parm.to_dict()
    return best_parm

def final_training_agents(env_params, models, path='plots/', **kwargs):
    """Trains the final agents with the best parameters."""
    train_results_dict = {}
    parm_dict = choose_parms()
    if kwargs:
        #add key and values from kwargs to parm_dict
        parm_dict.update(kwargs['kwargs'])

    for model in tqdm(models):
        parm_dict['rewards'] = model
        _, _, actions_callback = run_all(env_params, **parm_dict)
        train_results_dict[model] = actions_callback
    
    grouped_actions = [[np.mean(train_results_dict[model].group0_actions), 
        np.mean(train_results_dict[model].group1_actions)] for model in models]

    PLOT_PATH = path
    if 'model_name' in kwargs['kwargs']:
        PLOT_PATH = 'plots/'+kwargs['kwargs']['model_name']+'/'  

    grouped_barplot(grouped_actions, ['Group 0', 'Group 1'],
                        'Mean positive actions per group and model', 'Model',
                        'Mean positive actions', models, PLOT_PATH=PLOT_PATH)


if __name__ == '__main__':
    LEARNING_STEPS = 50000

    parm_csv = pd.read_csv('parms_test/results.csv')
    parm_csv.columns = ['index', 'learning_rate', 'gamma', 'clip_range', 'n_steps', 'model', 'bank_cash']

    #Choose best parameters
    best_parm = parm_csv.loc[parm_csv['bank_cash'].idxmax()]
    print(f"Best param config: {best_parm}")
    best_parm = best_parm.drop('index')
    best_parm = best_parm.drop('bank_cash')
    best_parm = best_parm.drop('model')
    best_parm = best_parm.to_dict()
                
    final_training_agents(learning_steps=LEARNING_STEPS, show_plot=False, **best_parm)
