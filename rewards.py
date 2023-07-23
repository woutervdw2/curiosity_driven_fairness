# coding=utf-8
# Copyright 2022 The ML Fairness Gym Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python2, python3
"""Reward functions for ML fairness gym.

These transforms are used to extract scalar rewards from state variables.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Optional
import core
import numpy as np

import sys


class NullReward(core.RewardFn):
  """Reward is always 0."""

  # TODO(): Find a better type for observations than Any.
  def __call__(self, observation):
    del observation  # Unused.
    return 0

class ScalarDeltaReward(core.RewardFn):
  """Extracts a scalar reward from the change in a scalar state variable."""

  def __init__(self, dict_key, baseline=0):
    """Initializes ScalarDeltaReward.

    Args:
      dict_key: String key for the observation used to compute the reward.
      baseline: value to consider baseline when first computing reward delta.
    """
    self.baseline = baseline
    self.dict_key = dict_key
    self.last_val = float(baseline)
    self.history = []

  # TODO(): Find a better type for observations than Any.
  def __call__(self, observation):
    """Computes a scalar reward from observation.

    The scalar reward is computed from the change in a scalar observed variable.

    Args:
      observation: A dict containing observations.
    Returns:
      scalar reward.
    Raises:
      TypeError if the observed variable indicated with self.dict_key is not a
        scalar.
    """

    #Debug info
    # print(f"""\n reward observation: {float(observation[self.dict_key])}\n
    # dict_key: {self.dict_key}\n
    # last_val: {self.last_val}""")

    # Validates that the state variable is a scalar with this float() call.

    if observation[self.dict_key].size != 1:
      current_val = np.sum(observation[self.dict_key])/4
      current_val = float(current_val)
      retval = current_val
    else:
      # Validates that the state variable is a scalar with this float() call.
      current_val = float(observation[self.dict_key])
  
      retval = current_val - self.last_val
    
    self.last_val = current_val
    self.history.append(retval)
    return retval
  
  def __reset__(self):
    self.last_val = float(self.baseline)

class ScalarDeltaRewardWithUCB(core.RewardFn):
  def __init__(self, dict_key, baseline=0, c=1.0, null=False):
    """Initializes ScalarDeltaReward.

    Args:
      dict_key: String key for the observation used to compute the reward.
      baseline: value to consider baseline when first computing reward delta.
    """
    self.baseline = baseline
    self.dict_key = dict_key
    self.last_val = float(baseline)
    self.c = c
    self.history = []
    #History of reward values [retval, UCB]
    self.value_history = [[], []]
    self.null = null

  # TODO(): Find a better type for observations than Any.
  def __call__(self, observation):
    """Computes a scalar reward from observation.

    The scalar reward is computed from the change in a scalar observed variable.

    Args:
      observation: A dict containing observations.
    Returns:
      scalar reward.
    Raises:
      TypeError if the observed variable indicated with self.dict_key is not a
        scalar.
    """

    #Debug info
    # print(f"""\n reward observation: {float(observation[self.dict_key])}\n
    # dict_key: {self.dict_key}\n

    # last_val: {self.last_val}""")
    if observation[self.dict_key].size != 1:
      current_val = np.sum(observation[self.dict_key])/4
      current_val = float(current_val)
      retval = current_val
      self.last_val = current_val
    else:
      # Validates that the state variable is a scalar with this float() call.
      current_val = float(observation[self.dict_key])
    
      retval = current_val - self.last_val
      self.last_val = current_val

    UCB = self._calculateUCB()
    self.value_history[0].append(retval)
    self.value_history[1].append(UCB)
    total_reward = retval + UCB
    if self.null:
      total_reward = self.null
    return total_reward
  
  def __reset__(self):
    self.last_val = self.baseline
    self.history = []
  
  def _update_history(self, action):
    #Update history with action
    self.history.append(action)
  
  def _calculateUCB(self):
    """Calculate the UCB value for the current history"""
    #Find last action
    last_action = self.history[-1]
    #Find number of times last action was taken
    if type(last_action) == np.ndarray:
      num_last_action = sum(np.array_equal(last_action, action) for action in self.history)
    else:
      num_last_action = self.history.count(last_action)


    #Calculate UCB
    UCB = self.c * np.sqrt(np.log(len(self.history))/num_last_action)
    return UCB

class ScalarDeltaRewardVisitCounts(core.RewardFn):
  def __init__(self, dict_key, baseline=0, beta=1.0, null=False):
    """Initializes ScalarDeltaReward.

    Args:
      dict_key: String key for the observation used to compute the reward.
      baseline: value to consider baseline when first computing reward delta.
    """
    self.baseline = baseline
    self.dict_key = dict_key
    self.last_val = float(baseline)
    self.beta = beta
    self.history = {}
    self.visit_count = None
    self.null = null
    #History of reward values [retval, value count]
    self.value_history = [[], []]
  # TODO(): Find a better type for observations than Any.
  def __call__(self, observation):
    """Computes a scalar reward from observation.

    The scalar reward is computed from the change in a scalar observed variable.

    Args:
      observation: A dict containing observations.
    Returns:
      scalar reward.
    Raises:
      TypeError if the observed variable indicated with self.dict_key is not a
        scalar.
    """
    
    #Debug info
    # print(f"""\n reward observation: {float(observation[self.dict_key])}\n
    # dict_key: {self.dict_key}\n

    # last_val: {self.last_val}""")
    if observation[self.dict_key].size != 1:
      current_val = np.sum(observation[self.dict_key])/4
      current_val = float(current_val)
      retval = current_val
    else:
      # Validates that the state variable is a scalar with this float() call.
      current_val = float(observation[self.dict_key])

      retval = current_val - self.last_val
      
    self.last_val = current_val

    self.value_history[0].append(retval)
    self.value_history[1].append(self.visit_count)
    total_reward = retval + self.visit_count
    if self.null:
      total_reward = self.visit_count
    return total_reward
  
  def __reset__(self):
    self.last_val = self.baseline
    self.history = {}
  
  def _update_history(self, state, action):
    """Update history with action state pair count"""
    if type(state) == np.ndarray:
      key = str(state) + str(action)
    else:
      key = str(state.group) + str(action)
    if key in self.history:
      self.history[key] += 1
    else:
      self.history[key] = 1
    
    self.visit_count = self._update_visit_count(key)
  
  def _update_visit_count(self, key):
    """Update the current visit count for state action pair"""

    visit_count = self.beta/np.sqrt(self.history[key])

    return visit_count
    

class BinarizedScalarDeltaReward(ScalarDeltaReward):
  """Extracts a binary reward from the sign of the change in a state variable."""

  # TODO(): Find a better type for observations than Any.
  def __call__(self, observation):
    """Computes binary reward from state.

    Args:
      observation: A dict containing observations.
    Returns:
      1 - if the state variable has gone up.
      0 - if the state variable has gone down.
      None - if the state variable has not changed.
    Raises:
      TypeError if the state variable indicated with self.dict_key is not a
        scalar.
    """

    delta = super(BinarizedScalarDeltaReward, self).__call__(observation)
    # Validate that delta is a scalar.
    _ = float(delta)
    if delta == 0:
      return None
    return int(delta > 0)


class VectorSumReward(core.RewardFn):
  """Extracts scalar reward that is the sum of a vector state variable.

  e.g.if state.my_vector = [1, 2, 4, 6], then
  VectorSumReward('my_vector')(state) returns 13.
  """

  def __init__(self, dict_key):
    """Initializes VectorSumReward.

    Args:
      dict_key: String key for the state variable used to compute the reward.
    """
    self.dict_key = dict_key

  # TODO(): Find a better type for observations than Any.
  def __call__(self, observation):
    """Computes scalar sum reward from state.

    Args:
      observation: An observation containing dict_key.
    Returns:
      Scalar sum of the vector observation defined by dict_key.
    Raises:
      ValueError if the dict_key is not in the observation.
    """
    if self.dict_key not in observation:
      raise ValueError("dict_key %s not in observation" % self.dict_key)
    return np.sum(observation[self.dict_key])
