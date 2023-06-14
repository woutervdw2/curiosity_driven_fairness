from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import logging
from typing import Any, Callable, List, Mapping, Optional, Text, Union

import attr
import core
import params
from agents import threshold_policies
import gym
import os

import rewards

import numpy as np
import tensorflow as tf


from stable_baselines3 import PPO
from stable_baselines3.common.policies import obs_as_tensor
import torch

@attr.s
class RlAgent(core.Agent):
    """Simple Reinforcement Learning agent"""

    observation_space = attr.ib()
    reward_fn = attr.ib()
    action_space = attr.ib(
        factory=lambda: gym.spaces.Discrete(2))
    rng = attr.ib(factory=np.random.RandomState)
    model = attr.ib(None)
    model_kind = attr.ib("ppo_lending/")
    model_name = attr.ib("UCB")


    #Check if model exists
    def __attrs_post_init__(self):
        print(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))+"/models/"+self.model_name)
        print(self.model_name)
        try:
            self.path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))+"/models/"+self.model_name
            print(self.path)
            self.load_model()
        except Exception as e:
            print("except:", e)
            print("except:", self.model_name)
            self.path = "/home/woutervdw2/Documents/thesis/code/ml-fairness-gym/curiosity/ppo_lending/models/"+self.model_name[:-10]
        try:
            print(f"Model path: {self.path}")
            self.load_model()
            print("Model loaded")
            self.model.set_parameters(load_path_or_dict=(self.path))
            print("Parameters set")
        except Exception as e:
            print("Model not found, try different model name or kind")
            print("error: ", e)
            exit()
        
        if self.reward_fn is None:
            self.reward_fn = rewards.NullReward()
        

    #function to load model from file
    def load_model(self):
        self.model = PPO.load(self.path)

    def _act_impl(self, observation, reward, done):
        """Returns an action based on the observation."""
        action, _ = self.model.predict(observation, deterministic=True)
        return action
    
    def predict_proba(self, obs):
        obs = self.transform_dict_values(obs)
        # obs = obs_as_tensor(obs, self.model.policy.device)
        dis = self.model.policy.get_distribution(obs)
        probs = dis.distribution.probs
        probs_np = probs.detach().cpu().numpy()
        return probs_np
    
    def transform_dict_values(self, dictionary):
        transformed_dict = {}
        for key, value in dictionary.items():
            if isinstance(value, np.ndarray):
                transformed_dict[key] = torch.as_tensor(value, device='cuda').reshape(1, -1)
            elif isinstance(value, dict):
                transformed_dict[key] = self.transform_dict_values(value)
            else:
                transformed_dict[key] = value
        return transformed_dict

