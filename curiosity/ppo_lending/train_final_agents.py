# #Add parent folder
import sys
sys.path.append("/home/woutervdw2/Documents/thesis/code/ml-fairness-gym")

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


def choose_parms(absolute=False):
    """Chooses the best parameters from the parm testing file."""
    parm_csv = pd.read_csv('/home/woutervdw2/Documents/thesis/code/ml-fairness-gym/parms_test/results.csv')
    parm_csv.columns = ['index', 'learning_rate', 'gamma', 'clip_range', 'n_steps', 'model', 'bank_cash']

    if absolute:
        #Choose best parameters absolute
        best_parm = parm_csv.loc[parm_csv['bank_cash'].idxmax()]
        best_parm = best_parm.drop('index')
        best_parm = best_parm.drop('bank_cash')
        best_parm = best_parm.drop('model')
        best_parm = best_parm.to_dict()
    else:
        #Choose best parameters relative start with highest bank cash
        parm_list = ['learning_rate', 'gamma', 'clip_range', 'n_steps']
        best_parm = {}
        for parm in parm_list:
            best = parm_csv.groupby(parm).mean().sort_values(by='bank_cash', ascending=False).index[0]
            best_parm[parm] = best
    print(f"Best param config: {best_parm}")
    return best_parm
  

def final_training_agents(env_params, models, path='plots/', **kwargs):
    """Trains the final agents with the best parameters."""
    train_results_dict = {}
    parm_dict = choose_parms()

    if kwargs:
        #add key and values from kwargs to parm_dict
        parm_dict.update(kwargs['kwargs'])

    for model in tqdm(models):
        print(f"Training {model}")
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
    print(choose_parms())
