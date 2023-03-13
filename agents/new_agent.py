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

@attr.s
class RlAgent(core.Agent):
    """Simple Reinforcement Learning agent"""

    observation_space = attr.ib()
    reward_fn = attr.ib()
    action_space = attr.ib(
        factory=lambda: gym.spaces.Discrete(2))
    rng = attr.ib(factory=np.random.RandomState)

    def _act_impl(self, observation, reward, done):
        action = self.action_space.sample()
        print(f"""\n 
        observation: {self.flatten_features(observation)}\n
        reward: {reward}\n
        action: {action}""")
        return action
