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

import numpy as np
import tensorflow as tf


from stable_baselines3 import PPO

@attr.s
class RlAgent(core.Agent):
    """Simple Reinforcement Learning agent"""

    observation_space = attr.ib()
    reward_fn = attr.ib()
    action_space = attr.ib(
        factory=lambda: gym.spaces.Discrete(2))
    rng = attr.ib(factory=np.random.RandomState)
    model = attr.ib(None)
    model_name = attr.ib("UCB")
    model_kind = attr.ib("ppo_lending")
    path = attr.ib("models/"+model_name+'/'+model_kind+'_'+model_name)

    #Check if model exists
    def __attrs_post_init__(self):
        try:
            self.load_model()
        except:
            print("Model not found, try different model name or kind")

    #function to load model from file
    def load_model(self):
        self.model = PPO.load(self.path)

    def _act_impl(self, observation, reward, done):
        """Returns an action based on the observation."""
        action, _ = self.model.predict(observation)
        return action


