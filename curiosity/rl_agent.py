# #Add parent folder
import sys
sys.path.append('../ml-fairness-gym')


#Lending environment without max bank cash
# from environments import lending
#Lending environment with max bank cash
from environments import lending_max3x as lending

from environments import lending_params
import rewards as rewards_fn
from gym import Env, spaces

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

import numpy as np
import matplotlib.pyplot as plt


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

    def _on_step(self) -> bool:
        #Save the actions and rewards per group
        group = np.argmax(a=self.locals['obs_tensor']['group'][0].cpu(), axis=0)
        if group == 0:
            self.group0_actions.append(self.locals['actions'])
            self.group0_rewards.append(self.locals['rewards'])
        else:
            self.group1_actions.append(self.locals['actions'])
            self.group1_rewards.append(self.locals['rewards'])

        if (self.num_timesteps % self.n_steps == 0) and self.verbose:
            self._training_summary()

    #Function that prints mean of actions and rewards per group in last 1000 steps
    def _training_summary(self) -> bool:
        group0_actions = np.array(self.group0_actions)
        group1_actions = np.array(self.group1_actions)
        group0_rewards = np.array(self.group0_rewards)
        group1_rewards = np.array(self.group1_rewards)
        print('mean actions:', np.mean(group0_actions[-self.n_steps:]), np.mean(group1_actions[-self.n_steps:]))
        print('mean rewards:', np.mean(group0_rewards[-self.n_steps:]), np.mean(group1_rewards[-self.n_steps:]))

        return True


#function to initialize the environment
def init_env(env_params, rewards='scalar'):
    env = lending.DelayedImpactEnv(env_params)
    if rewards == 'scalar':
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
                    c=1.0)
    elif rewards == 'visit_count':
        env.reward_fn = rewards_fn.ScalarDeltaRewardVisitCounts(
                    'bank_cash',
                    baseline=env.initial_params.bank_starting_cash)
    else:
        print('Reward function not recognized')
        sys.exit()
    
    print('Test environment')
    #Check environment is working
    check_env(env, warn=True)
    print('Environment is working')

    return env

#function to train the agent
def train_agent(env, learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, clip_range=0.2,
                seed=None, learning_steps=100000, model_name='ppo_lending',
                  path='models/', rewards='scalar', verbose=1):
    #Create the agent
    agent = PPO('MultiInputPolicy', env, verbose=1, learning_rate=learning_rate,
                 n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, gamma=gamma,
                   clip_range=clip_range, seed=seed)

    #Create the callback
    #Output action and reward summary every 10% of the learning steps
    actions_callback = SaveActionsCallback(n_steps=np.round(learning_steps*0.1), verbose=verbose)

    # Train the agent
    agent.learn(total_timesteps=learning_steps,
                progress_bar=True, callback=actions_callback)

    # Save the agent
    agent.save(path+model_name+'_'+rewards)

    return agent, actions_callback

#function to test the agent
def test_agent(agent, env, n_steps=1000):
    # Test the trained agent
    obs = env.reset()
    test_rewards = []
    test_actions = []
    for _ in range(n_steps):
        # print('before:', obs)
        action, __ = agent.predict(obs)
        # print('action:', action)
        test_actions.append(action)
        obs, rewards, done, info = env.step(action)
        # print('after:', obs)
        # print('reward:', rewards)
        test_rewards.append(rewards)
        if done:
            obs = env.reset()

    print(f"""Test results:\n
    Bank cash: {obs['bank_cash']}\n
    Average test reward: {np.mean(test_rewards)}
    """)

    test_results = {'bank_cash': obs['bank_cash'],
                    'avg_test_reward': np.mean(test_rewards),
                    'test_rewards': test_rewards}
    
    return test_results

#function to get baseline agent results
def get_baseline_results(env, n_steps=1000):
    # Test the trained agent
    obs = env.reset()
    baseline_rewards = []
    baseline_actions = []
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
            obs = env.reset()

    print(f"""Baseline results:\n
    Bank cash: {obs['bank_cash']}\n
    Average baseline rewards: {np.mean(baseline_rewards)}""")

    baseline_results = {'bank_cash': obs['bank_cash'],
                        'avg_baseline_reward': np.mean(baseline_rewards),
                        'baseline_rewards': baseline_rewards}
    
    return baseline_results

#function to plot the results
def plot_results(actions_callback, model_name='ppo_lending', path='models/', rewards='scalar', show_plot=True):
    group0_actions = np.array(actions_callback.group0_actions)
    group1_actions = np.array(actions_callback.group1_actions)


    #Barplot of positive actions per group
    fig = plt.figure()
    plt.bar(x=['group0', 'group1'], height=[np.mean(group0_actions), np.mean(group1_actions)])
    plt.title('Average positive actions per group with '+rewards+' rewards')
    plt.savefig(path+model_name+rewards+'_average_positive_actions.png')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    #Barplot of mean rewards per group
    fig = plt.figure()
    plt.bar(x=['group0', 'group1'], height=[np.mean(actions_callback.group0_rewards),
                                             np.mean(actions_callback.group1_rewards)])
    plt.title('Average rewards per group with '+rewards+' rewards')
    plt.savefig(path+model_name+rewards+'_average_rewards.png')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    #Show barplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Average positive actions and rewards per group with '+rewards+' rewards')
    ax1.bar(x=['group0', 'group1'], height=[np.mean(group0_actions), np.mean(group1_actions)])
    ax1.set_title('Average positive actions')
    ax2.bar(x=['group0', 'group1'], height=[np.mean(actions_callback.group0_rewards),
                                                np.mean(actions_callback.group1_rewards)])
    ax2.set_title('Average rewards with '+rewards+' rewards')
    plt.savefig(path+model_name+rewards+'_average_positive_actions_rewards.png')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

#Function to run everything
def run_all(env_params, learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, clip_range=0.2,
            seed=None, learning_steps=100000, model_name='ppo_lending', verbose=1,
            path='models/', n_test_steps=1000, rewards='scalar', show_plot=True):
    
    path = path+rewards+'/'
    #Initialize environment
    env = init_env(env_params, rewards=rewards)

    #Train agent
    print('Training agent...')
    agent, actions_callback = train_agent(env, learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size,
                                          n_epochs=n_epochs, gamma=gamma, clip_range=clip_range, seed=seed,
                                          learning_steps=learning_steps, model_name=model_name, path=path, 
                                          rewards=rewards, verbose=verbose)
    print('Agent trained')

    #Test agent
    print('Testing agent...')
    test_results = test_agent(agent, env, n_steps=n_test_steps)
    print('Agent tested')

    #Get baseline results
    print('Getting baseline results...')
    baseline_results = get_baseline_results(env, n_steps=n_test_steps)
    print('Baseline results obtained')

    #Plot results
    print('Plotting results...')
    plot_results(actions_callback, model_name=model_name, path=path, rewards=rewards, show_plot=show_plot)
    print('Results plotted')

    return test_results, baseline_results, actions_callback


if __name__ == '__main__':
    #Define environment parameters
    group_0_prob = 0.5
    bank_starting_cash = np.float32(10000)
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

# group_0_prob = 0.5
# bank_starting_cash = np.float32(10000)
# interest_rate = 1.0
# cluster_shift_increment = 0.01
# cluster_probabilities = lending_params.DELAYED_IMPACT_CLUSTER_PROBS

# env_params = lending_params.DelayedImpactParams(
#         applicant_distribution=lending_params.two_group_credit_clusters(
#             cluster_probabilities= cluster_probabilities,
#             group_likelihoods=[group_0_prob, 1 - group_0_prob]),
#         bank_starting_cash=bank_starting_cash,
#         interest_rate=interest_rate,
#         cluster_shift_increment=cluster_shift_increment,
#     )

# run_all(env_params, learning_steps=1000, rewards='visit_count', show_plot=True)




