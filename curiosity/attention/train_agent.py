# #Add parent folder
import sys
sys.path.append("/home/woutervdw2/Documents/thesis/code/ml-fairness-gym")

import os

from environments import attention_allocation_copy as attention_allocation

import rewards as rewards_fn
from gym import Env, spaces

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter, vec_env
from stable_baselines3.common.logger import configure

import pickle

import numpy as np
import matplotlib.pyplot as plt


    


def init_env(env_params, rewards='scalar', beta=1, c=1, test=False):
    if not test:
        env = attention_allocation.LocationAllocationEnv(env_params)
    else:
        env = attention_allocation.LocationAllocationEnv(env_params, test=test)
    
    if (rewards == 'scalar' or test):
        if not test:
            print("Using scalar reward")
        else:
            print("Using scalar reward for test")
            
        env.reward_fn = rewards_fn.ScalarDeltaReward(
                'incidents_seen')
    elif rewards == 'UCB':
        print("Using UCB reward")
        env.reward_fn = rewards_fn.ScalarDeltaRewardWithUCB(
                    'incidents_seen',
                    c=c)
    elif rewards == 'visit_count':
        print("Using visit count reward")
        env.reward_fn = rewards_fn.ScalarDeltaRewardVisitCounts(
                    'incidents_seen',
                    beta=beta)
    else:   
        print("Invalid reward function")
        sys.exit()
    
    print("Test environment")
    check_env(env, warn=True)
    print("Environment is valid")
    
    env.reset()
    
    return env

def train_agent(env, path, c, reward='scalar', learning_rate=0.0003, n_steps=1024, batch_size=64, n_epochs=1, gamma=0.99, clip_range=0.2,
                seed=None, learning_steps=100000, verbose=1, test_env=None, save=True):
    
    agent = PPO('MultiInputPolicy', env, verbose=verbose, learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, gamma=gamma, clip_range=clip_range, seed=seed)

    save_path = path+reward+"_"+str(c)+"/"
    print("Start learning")
    if test_env:
        eval_callback = EvalCallback(test_env, best_model_save_path=save_path+reward+'_best', log_path=save_path+reward+'_logs', eval_freq=max(np.round(learning_steps*0.1), 100), deterministic=True, render=False, verbose=1)
        agent.learn(total_timesteps=learning_steps, progress_bar=True, callback=eval_callback)
    else:
        agent.learn(total_timesteps=learning_steps, progress_bar=True)
    print("Finished learning")
    if save:
        agent.save(f"{save_path}{reward}")
    
    return agent

def plot_cumulative_reward(agent, rewards, path, c, show_plot=True):
    reward_fn = agent.reward_fn
    save_path = path+rewards+"_"+str(c)+"/"
    if rewards == 'scalar':
        plt.plot(np.cumsum(reward_fn.history), label='Reward')
    elif rewards == 'UCB':
        plt.plot(np.cumsum(reward_fn.value_history[0]), label='Reward')
        plt.plot(np.cumsum(reward_fn.value_history[1]), label='UCB')
    elif rewards == 'visit_count':
        plt.plot(np.cumsum(reward_fn.value_history[0]), label='Reward')
        plt.plot(np.cumsum(reward_fn.value_history[1]), label='Visit count')
    plt.title(f'Cumulative reward history with {rewards} rewards')
    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig(save_path+'cumulative_reward_history.png')
    if show_plot:
        plt.show()
    else:
        plt.close()
        
def main():
    LEARNING_STEPS = 600000
    MODELS = ['visit_count', 'scalar', 'UCB']
    curiosities = [3, 6]
    env_params = attention_allocation.Params(
        n_locations = 6,
        prior_incident_counts =  (600, 500, 400, 300, 200, 100),
        incident_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        n_attention_units = 4,
        miss_incident_prob = (0., 0., 0., 0., 0., 0.),
        extra_incident_prob = (0., 0., 0., 0., 0., 0.),
        dynamic_rate = 0.1
    )
    
    for c in curiosities:
        if c == 3:
            MODELS = ['UCB']
        else:
            MODELS = ['visit_count', 'scalar', 'UCB']
        for model in MODELS:
            print("Start training model: ", model, " with curiosity: ", c)
            env = init_env(env_params, rewards=model, beta=1, c=c, test=False)
            test_env = init_env(env_params, rewards=model, beta=1, c=c, test=True)
            #Train agent
            path = '../../models/allocation/'
            if not os.path.exists(path):
                os.makedirs(path)
            
            print("start training")
            train_agent(env, path, c=c, reward=model, learning_rate=0.0003, n_steps=1024, batch_size=64, n_epochs=10, gamma=0.99, clip_range=0.2,
                        seed=None, learning_steps=LEARNING_STEPS, verbose=0, test_env=test_env)
            
            #Plot cumulative reward
            plot_cumulative_reward(env, model, path, c, show_plot=False)

if __name__ == '__main__':
    main()
    
    
    # env_params = attention_allocation.Params(
    #     n_locations = 5,
    #     prior_incident_counts =  (500, 400, 300, 200, 100),
    #     incident_rates = [0.1, 0.2, 0.3, 0.4, 0.5],
    #     n_attention_units = 4,
    #     miss_incident_prob = (0., 0., 0., 0., 0.),
    #     extra_incident_prob = (0., 0., 0., 0., 0.),
    #     dynamic_rate = 0.0    
    # )
    
    # env = init_env(env_params, rewards='visit_count', beta=1, c=1, test=False)
    # test_env = init_env(env_params, rewards='visit_count', beta=1, c=1, test=True)
    # #Train agent
    # path = '../../models/allocation/'
    # if not os.path.exists(path):
    #     os.makedirs(path)
    
    # print("start training")
    # train_agent(env, path, c=1, reward='visit_count', learning_rate=0.0003, n_steps=1024, batch_size=64, n_epochs=10, gamma=0.99, clip_range=0.2,
    #             seed=None, learning_steps=100, verbose=1, test_env=test_env)
    
    