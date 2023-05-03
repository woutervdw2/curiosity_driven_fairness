"""File that trains agents on chosen parm combination, 
Test the agents on the test set and plots the results.
Last it assessess the fairness of the agents on the lending environment"""

import sys
sys.path.append('../ml-fairness-gym')

#Lending environment without max bank cash
# from environments import lending
#Lending environment with max bank cash
from environments import lending_max3x as lending

from environments import lending_params

from train_final_agents import final_training_agents
from compare_agents import compare_agents
from assess_models import main as assess_models

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    #Constants
    LEARNING_STEPS = [100000, 200000, 300000]
    MODELS = ['visit_count', 'scalar', 'UCB']
    curiosity = [0.1, 1, 2]
    for c in curiosity:
        if c==0.1:
            LEARNING_STEPS = [200000, 300000]
        else:
            LEARNING_STEPS = [100000, 200000, 300000]
        for n_steps in LEARNING_STEPS:
            model_name = f'ppo_lending/{n_steps}_{c}_final'
            #Initialize environment
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
                    cluster_shift_increment=cluster_shift_increment
                )
            
            # run_all arguments
            train_args = {
                'verbose': 0,
                'learning_steps': n_steps,
                'show_plot': True,
                'model_name': model_name,
                'beta': c,
                'c': c
            }

            #While training print terminal output colored red
            sys.stdout.write("\033[1;31m")
            #Train final agents
            print("Training final agents"+'-'*100)
            final_training_agents(env_params, models=MODELS, kwargs=train_args)


            #Compare final agents
            test_args = {
                'n_test_steps': 1000,
                'train': False,
                'show_plot': False,
                'model_name': model_name
            }
            
            #While comparing print terminal output colored blue
            sys.stdout.write("\033[1;34m")
            print("Comparing agents on test environments"+'-'*100)
            compare_agents(env_params, models=MODELS, kwargs=test_args)

            #While assessing print terminal output colored yellow
            sys.stdout.write("\033[1;33m")
            print("Assessing models"+'-'*100)
            for model in MODELS:
                assess_models(reward=model, model_name=model_name)


if __name__ == "__main__":
    main()