# #Add parent folder
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

#Lending environment without max bank cash
# from environments import lending
#Lending environment with max bank cash
from environments import lending_max3x as lending

from environments import lending_params
import rewards as rewards_fn
from gym import Env, spaces

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.logger import configure

import pickle

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



# #Callback saving the actions per group over the course of training
class SaveActionsCallback(BaseCallback):
    def __init__(self, n_steps=1000, verbose=1):
        super(SaveActionsCallback, self).__init__(n_steps)
        self.group0_actions = []
        self.group1_actions = []
        self.group0_rewards = []
        self.group1_rewards = []
        self.n_steps = int(n_steps)
        self.verbose = verbose
        self.batch_regret = []
        self.regret = []

        self.group1_regret = []
        self.group0_regret = []
        self.group1_batch_regret = []
        self.group0_batch_regret = []

        self.group0_default = []
        self.group1_default = []
    
        

    def _on_step(self) -> bool:
        #Save the actions and rewards per group
        state = self.locals['env'].get_attr('state')[0]
        group = np.argmax(a=self.locals['obs_tensor']['group'][0].cpu(), axis=0)
        if group == 0:
            self.group0_actions.append(self.locals['actions'])
            self.group0_rewards.append(self.locals['rewards'])
            self.group0_default.append(state.will_default)
        else:
            self.group1_actions.append(self.locals['actions'])
            self.group1_rewards.append(self.locals['rewards'])
            self.group1_default.append(state.will_default)

        if (self.num_timesteps % self.n_steps == 0) and self.verbose:
            self._training_summary()
        
        self.compute_regret()
    

    def _on_rollout_end(self) -> None:
        self.regret.append(np.sum(self.batch_regret))
        self.group0_regret.append(np.sum(self.group0_batch_regret))
        self.group1_regret.append(np.sum(self.group1_batch_regret))

        self.group1_batch_regret = []
        self.group0_batch_regret = []
        self.batch_regret = []
        

    def compute_regret(self):       
        action = self.locals['actions'][0]
        group = np.argmax(a=self.locals['obs_tensor']['group'][0].cpu(), axis=0)
        applicant_features = np.argmax(a=self.locals['obs_tensor']['applicant_features'][0].cpu(), axis=0)
        state = self.locals['env'].get_attr('state')[0]
        P_will_default = state.params.applicant_distribution.components[group].components[applicant_features].will_default.p
        P_paying_back = 1-P_will_default

        #Compute optimal action
        expected_rejected = 0
        expected_accepted = -1*(P_will_default * state.params.loan_amount) + (P_paying_back * state.params.interest_rate)
        best_action = max(expected_rejected, expected_accepted)
        if action == 1:
            regret = best_action - expected_accepted
        else:
            regret = best_action - expected_rejected
        self.batch_regret.append(regret)
        if group == 1:
            self.group1_batch_regret.append(regret)
        else:
            self.group0_batch_regret.append(regret)


   
    #Function that prints mean of actions and rewards per group in last 1000 steps
    def _training_summary(self) -> bool:
        group0_actions = np.array(self.group0_actions)
        group1_actions = np.array(self.group1_actions)
        group0_rewards = np.array(self.group0_rewards)
        group1_rewards = np.array(self.group1_rewards)
        print('mean actions:', np.mean(group0_actions[-self.n_steps:]), np.mean(group1_actions[-self.n_steps:]))
        print('mean rewards:', np.mean(group0_rewards[-self.n_steps:]), np.mean(group1_rewards[-self.n_steps:]))

        return True
    
    def __getstate__(self):
        """Return class attributes to be pickled."""
        state = self.__dict__.copy()
        del state["training_env"]
        del state["locals"]
        del state["globals"]
        del state["parent"]
        del state["model"]
        del state["logger"]
        return state
    



#function to initialize the environment
def init_env(env_params, rewards='scalar', beta=1, c=1, test=False):
    if not test:
        env = lending.DelayedImpactEnv(env_params)
    else:
        env = lending.DelayedImpactEnv(env_params, test=test)
    if (rewards == 'scalar' or test):
        env.reward_fn = rewards_fn.ScalarDeltaReward(
                    'bank_cash', 
                    baseline=env.initial_params.bank_starting_cash)
    elif rewards == 'binary':
        env.reward_fn = rewards_fn.BinaryDeltaReward(
                    'bank_cash',
                    baseline=env.initial_params.bank_starting_cash)
    elif rewards == 'UCB':
        env.reward_fn = rewards_fn.ScalarDeltaRewardWithUCB(
                    'bank_cash',
                    baseline=env.initial_params.bank_starting_cash,
                    c=c)
    elif rewards == 'visit_count':
        env.reward_fn = rewards_fn.ScalarDeltaRewardVisitCounts(
                    'bank_cash',
                    baseline=env.initial_params.bank_starting_cash,
                    beta=beta)
    else:
        print('Reward function not recognized')
        sys.exit()
    
    env.reset()
    
    print('Test environment')
    #Check environment is working
    check_env(env, warn=True)
    print('Environment is working')

    env.reset()
    return env

#function to train the agent
def train_agent(env, path, reward='scalar', learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, clip_range=0.2,
                seed=None, learning_steps=100000, verbose=1, test_env=None):
    

    #Create the agent
    agent = PPO('MultiInputPolicy', env, learning_rate=learning_rate,
                 n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, gamma=gamma,
                   clip_range=clip_range, verbose=verbose, seed=seed)


    #Create the callbackS
    #Output action and reward summary every 10% of the learning steps
    actions_callback = SaveActionsCallback(n_steps=np.round(learning_steps*0.1), verbose=verbose)

    test_env = test_env
    #Eval callback that tests the agent every 10% of the learning steps
    eval_callback = EvalCallback(test_env, best_model_save_path=path+reward+'_best',
                                    log_path=path+reward+'_logs', eval_freq=max(np.round(learning_steps*0.1), 100),
                                    deterministic=True, render=False, verbose=1)  
    
    callback = CallbackList([actions_callback, eval_callback])

    # Train the agent
    agent.learn(total_timesteps=learning_steps,
                progress_bar=True, callback=callback)

    # Plot the results
    results_plotter.plot_results([path], 10e6, results_plotter.X_TIMESTEPS,
                                  "PPO Lending")
    plt.savefig(path+reward+'_result_monitor.png')
    plt.show()
    # Save the agent
    agent.save(path+reward)

    #Save actions callback to pickle
    with open(path+reward+'_actions_callback.pkl', 'wb') as f:
        pickle.dump(actions_callback, f)

        
    return agent, actions_callback

# function to test the agent
def test_agent(agent, env, n_steps=100, nr_runs=10):
    # Test the trained agent
    test_rewards = []
    test_actions = []
    bank_cash_history = []
    group1_actions = []
    group2_actions = []
    for _ in tqdm(range(nr_runs)):
        obs = env.reset()
        for _ in range(n_steps):
            # print('before:', obs)
            action, __ = agent.predict(obs, deterministic=True)
            # print('action:', action)
            test_actions.append(action)

            if env.state.group_id == 0:
                group1_actions.append(action)
            else:
                group2_actions.append(action)
            obs, rewards, done, info = env.step(action)
            # print('after:', obs)
            # print('reward:', rewards)
            test_rewards.append(rewards)
            if done:
                bank_cash_history.append(float(obs['bank_cash']))
                obs = env.reset()

        bank_cash_history.append(float(obs['bank_cash']))
    print(f"""Test results:\n
    Average bank cash: {np.mean(bank_cash_history)}\n
    Average test reward: {np.mean(test_rewards)}
    """)

    test_results = {'bank_cash': bank_cash_history,
                    'avg_test_reward': np.mean(test_rewards),
                    'test_rewards': test_rewards,
                    'group1_actions': group1_actions,
                    'group2_actions': group2_actions}
    
    return test_results


#function to get baseline agent results
def get_baseline_results(env, n_steps=100, nr_runs=10):
    # Test the trained agent
    baseline_rewards = []
    baseline_actions = []
    bank_cash_history = []
    
    for _ in range(nr_runs):
        obs = env.reset()
        for _ in range(n_steps):
            # print('before:', obs)
            action = env.action_space.sample()
            # print('action:', action)
            baseline_actions.append(action)
            obs, rewards, done, info = env.step(action)
            # print('after:', obs)
            # print('reward:', rewards)
            baseline_rewards.append(rewards)
            if done:
                bank_cash_history.append(float(obs['bank_cash']))
                obs = env.reset()
        bank_cash_history.append(float(obs['bank_cash']))

    print(f"""Baseline results:\n
    Average bank cash: {np.mean(bank_cash_history)}\n
    Average baseline rewards: {np.mean(baseline_rewards)}""")

    baseline_results = {'bank_cash': bank_cash_history,
                        'avg_baseline_reward': np.mean(baseline_rewards),
                        'baseline_rewards': baseline_rewards}
    
    return baseline_results

#function to plot the results
def plot_results(actions_callback, path='models/', rewards='scalar', show_plot=True):
    group0_actions = np.array(actions_callback.group0_actions)
    group1_actions = np.array(actions_callback.group1_actions)


    #Barplot of positive actions per group
    fig = plt.figure()
    plt.bar(x=['group0', 'group1'], height=[np.mean(group0_actions), np.mean(group1_actions)])
    plt.title('Average positive actions per group with '+rewards+' rewards')
    plt.savefig(path + '_average_positive_actions.png')
    if show_plot:
        plt.show()
    else:
        plt.close()


    #Barplot of mean rewards per group
    fig = plt.figure()
    plt.bar(x=['group0', 'group1'], height=[np.mean(actions_callback.group0_rewards),
                                             np.mean(actions_callback.group1_rewards)])
    plt.title('Average rewards per group with '+rewards+' rewards')
    plt.savefig(path+'/'+'_average_rewards.png')
    if show_plot:
        plt.show()
    else:
        plt.close()


    #Show barplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Average positive actions and rewards per group with '+rewards+' rewards')
    ax1.bar(x=['group0', 'group1'], height=[np.mean(group0_actions), np.mean(group1_actions)])
    ax1.set_title('Average positive actions')
    ax2.bar(x=['group0', 'group1'], height=[np.mean(actions_callback.group0_rewards),
                                                np.mean(actions_callback.group1_rewards)])
    ax2.set_title('Average rewards with '+rewards+' rewards')
    plt.savefig(path+'_average_positive_actions_rewards.png')
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_reward_progress(agent, rewards, path, show_plot=True):
    reward_fn = agent.reward_fn
    fig = plt.figure()
    if rewards == 'scalar':
        plt.plot(reward_fn.history, label='Reward')
    elif rewards == 'UCB':
        plt.plot(reward_fn.value_history[0], label='Reward')
        plt.plot(reward_fn.value_history[1], label='UCB')
    elif rewards == 'visit_count':
        plt.plot(reward_fn.value_history[0], label='Reward')
        plt.plot(reward_fn.value_history[1], label='Visit count')
    plt.title(f'Reward history with {rewards} rewards')
    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig(path+'reward_history.png')
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_cumulative_reward(agent, rewards, path, show_plot=True):
    reward_fn = agent.reward_fn
    fig = plt.figure()
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
    plt.savefig(path+'cumulative_reward_history.png')
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_cumulative_actions_per_group(rewards, action_callback, path, show_plot=True):
    fig = plt.figure()
    plt.plot(np.cumsum(action_callback.group0_actions), label='Group 0')
    plt.plot(np.cumsum(action_callback.group1_actions), label='Group 1')
    plt.title('Cumulative actions per group with '+rewards+' rewards')
    plt.xlabel('Timesteps')
    plt.ylabel('Actions')
    plt.legend()
    plt.savefig(path+'cumulative_actions_per_group.png')
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_regret(rewards, action_callback, path, show_plot=True):
    pass
#     fig = plt.figure()
#     plt.plot(action_callback.regret, label='Regret')
#     plt.title('Regret over time for '+rewards+' model')
#     plt.xlabel('Timesteps')
#     plt.ylabel('Regret')
#     #Set x tick labels such that every label for the list is 256 timesteps apart
#     nr_ticks = int(np.round(len(action_callback.regret)/7))
#     plt.xticks(ticks=np.arange(0, len(action_callback.regret), nr_ticks), labels=np.arange(256, (256*len(action_callback.regret)+1), nr_ticks*256))

#     plt.legend()
#     plt.savefig(path+'regret.png')
#     if show_plot:
#         plt.show()
#     else:
#         plt.close()

def plot_regret_per_group(rewards, action_callback, path, show_plot=True):
    pass
#     fig = plt.figure()
#     plt.plot(action_callback.group0_regret, label='Group 0')
#     plt.plot(action_callback.group1_regret, label='Group 1')
#     plt.title('Regret per group over time for '+rewards+' model')
#     plt.xlabel('Timesteps')
#     plt.ylabel('Regret')
#     nr_ticks = int(np.round(len(action_callback.group0_regret)/7))
#     plt.xticks(ticks=np.arange(0, len(action_callback.group0_regret), nr_ticks), labels=np.arange(256, (256*len(action_callback.group0_regret)+1), nr_ticks*256))


#     plt.legend()
#     plt.savefig(path+'regret_per_group.png')
#     if show_plot:
#         plt.show()
#     else:
#         plt.close()



#Function to run everything
def run_all(env_params, beta=1, c=1, learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, clip_range=0.2,
            seed=None, learning_steps=100000, model_name='ppo_lending/', verbose=1,
            path='models/', n_test_steps=100, rewards='scalar', show_plot=True, train=True):
    
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))+"/"+path+model_name+rewards+'/'
    os.makedirs(path, exist_ok=True)
    #Initialize environment
    env = init_env(env_params, rewards=rewards, beta=beta, c=c)


    #Train agent
    if train:
        test_env = init_env(env_params, rewards=rewards, beta=beta, c=c, test=True)
        test_env = Monitor(test_env, path, allow_early_resets=True)
        print('Training agent...')
        agent, actions_callback = train_agent(env, path, reward=rewards, learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size,
                                            n_epochs=n_epochs, gamma=gamma, clip_range=clip_range, seed=seed,
                                            learning_steps=learning_steps,
                                            verbose=verbose, test_env=test_env)
        print('Agent trained')
        plot_reward_progress(env, rewards, path, show_plot)
        plot_cumulative_reward(env, rewards, path, show_plot)
        plot_cumulative_actions_per_group(rewards, actions_callback, path, show_plot)
        plot_regret(rewards, actions_callback, path, show_plot)
        plot_regret_per_group(rewards, actions_callback, path, show_plot)
    else:
        #Load agent
        print('Loading agent...')
        agent = PPO.load(path+rewards+"_best/best_model", env=env)
        print('Agent loaded')

    #Test agent
    print('Testing agent...')
    test_results = test_agent(agent, env, n_steps=n_test_steps)
    print('Agent tested')

    #Get baseline results
    # print('Getting baseline results...')
    # baseline_results = get_baseline_results(env, n_steps=n_test_steps)
    # print('Baseline results obtained')

    #Plot results
    if train:
        print('Plotting results...')
        plot_results(actions_callback, path=path, rewards=rewards, show_plot=show_plot)
        print('Results plotted')

    if train:
        return test_results, baseline_results, actions_callback
    else:
        return test_results, None, None
    

if __name__ == '__main__':
    #Define environment parameters
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

    #Run everything
    run_all(env_params, learning_steps=1000, rewards='scalar', show_plot=True)





