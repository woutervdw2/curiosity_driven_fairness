# #Add parent folder
import sys
sys.path.append('../ml-fairness-gym')

from environments import lending
from environments import lending_params
import rewards
from gym import Env, spaces

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

import numpy as np
import matplotlib.pyplot as plt

import sys

model_name = 'ppo_lending'
PATH = f'../ml-fairness-gym/{model_name}/'

group_0_prob = 0.5
bank_starting_cash = np.float32(10000)
interest_rate = 0.5
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

env = lending.DelayedImpactEnv(env_params)
env.reward_fn = rewards.ScalarDeltaReward(
                'bank_cash', 
                baseline=env.initial_params.bank_starting_cash)


# Check the environment
check_env(env)

#test environment random actions
obs = env.reset()
rewards = []
for i in range(10):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    rewards.append(reward)
    if done:
        obs = env.reset()

# Create the agent
agent = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=
            PATH+"tensorboard/", device='cuda', n_steps=2048, batch_size=64)


#Callback saving the actions per group over the course of training
class SaveActionsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(SaveActionsCallback, self).__init__(verbose)
        self.group0_actions = []
        self.group1_actions = []
        self.group0_rewards = []
        self.group1_rewards = []

    def _on_step(self) -> bool:
        group = np.argmax(a=self.locals['obs_tensor']['group'][0].cpu(), axis=0)
        if group == 0:
            self.group0_actions.append(self.locals['actions'])
            self.group0_rewards.append(self.locals['rewards'])
        else:
            self.group1_actions.append(self.locals['actions'])
            self.group1_rewards.append(self.locals['rewards'])
        return True


actions_callback = SaveActionsCallback()

# Train the agent
agent.learn(total_timesteps=300000
            , log_interval=10, tb_log_name=model_name, 
            progress_bar=True, callback=actions_callback)

# Save the agent
agent.save(PATH+"ppo_lending")

# Load the agent
agent = PPO.load(PATH+"ppo_lending")

# Test the trained agent
obs = env.reset()
test_rewards = []
test_actions = []
for i in range(10):
    # print('before:', obs)
    action, _states = agent.predict(obs)
    # print('action:', action)
    test_actions.append(action)
    obs, rewards, done, info = env.step(action)
    # print('after:', obs)
    # print('reward:', rewards)
    test_rewards.append(rewards)
    if done:
        obs = env.reset()

print('final test:', obs)
print("Average test reward: ", np.mean(test_rewards))
print('test actions:', test_actions)

#baseline model
obs = env.reset()
baseline_rewards = []
for i in range(1000):
    # print('before:', obs)
    action = env.action_space.sample()
    # print('action:', action)
    obs, rewards, done, info = env.step(action)
    # print('after:', obs)
    # print('reward:', rewards)
    baseline_rewards.append(rewards)
    if done:
        obs = env.reset()

print('final baseline:', obs)
print("Average base reward: ", np.mean(baseline_rewards))


group0_actions = np.array(actions_callback.group0_actions)
group1_actions = np.array(actions_callback.group1_actions)

#Barplot of positive actions per group
plt.bar(x=['group0', 'group1'], height=[np.mean(group0_actions), np.mean(group1_actions)])
plt.title('Average positive actions per group')
plt.savefig(PATH+'average_positive_actions.png')
plt.show()

#Barplot of mean rewards per group
plt.bar(x=['group0', 'group1'], height=[np.mean(actions_callback.group0_rewards),
                                         np.mean(actions_callback.group1_rewards)])
plt.title('Average rewards per group')
plt.savefig(PATH+'average_rewards.png')
plt.show()









