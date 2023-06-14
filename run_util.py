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
"""Utilities for running and measuring gym simulations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import gin
import tqdm
import sys
import environments

import numpy as np
import os


FLAGS = flags.FLAGS

flags.DEFINE_boolean("use_tqdm", True,
                     "Use tqdm to visually represent progress in simulations.")


@gin.configurable
def run_simulation(env, agent, metrics, num_steps, seed=100, agent_seed=50):
  """Perform a simple simulation and return a measurement.

  Args:
    env: A `core.FairnessEnv`.
    agent: A `core.Agent`.
    metrics: A list of `core.Metric` instances, a dict of {name: `core.Metric`}
      or a single `core.Metric` instance.
    num_steps: An integer indicating the number of steps to simulate in each
      episode.
    seed: An integer indicating a random seed.
    agent_seed: An integer indicating a random seed for the agent.

  Returns:
    A list of measurements if multiple metrics else a single measurement for a
    single metric.
  """
  agent.seed(agent_seed)
  env.seed(seed)
  observation = env.reset()
  done = False

  print("Starting simulation")
  simulation_iterator = tqdm.trange if FLAGS.use_tqdm else range
  actions_group0 = []
  actions_group1 = []
  default0 = []
  default1 = []
  group0_features = []
  group1_features = []
  for epoch in simulation_iterator(num_steps):
    # Update the agent with any changes to the observation or action space.
    agent.action_space, agent.observation_space = (env.action_space,
                                                   env.observation_space)
    
    action = agent.act(observation, done)
    try:
      if env.state.group_id == 0:
        actions_group0.append(int(action))
        default0.append(bool(env.state.will_default))
        group0_features.append(np.argmax(env.state.applicant_features))
      else:
        actions_group1.append(int(action))
        default1.append(bool(env.state.will_default))
        group1_features.append(np.argmax(env.state.applicant_features))
    except AttributeError:
      pass
    # TODO(): Remove reward from this loop.
    observation, _, done, _ = env.step(action)
    if done:
      break
    
    try:
      with open(os.path.abspath(os.path.join(os.path.dirname(__file__)))+"/models/"+agent.model_name[:-10]+"sim_results.txt", "w") as f:
        f.write(f"""{actions_group0}\n
                {actions_group1}\n  
                {default0}\n
                {default1}\n
                {group0_features}\n
                {group1_features}""")
    except AttributeError:
      pass
    
  print("Measuring metrics")
  if isinstance(metrics, list):
    return [metric.measure(env) for metric in metrics]
  elif isinstance(metrics, dict):
    return {name: metric.measure(env) for name, metric in metrics.items()}
  else:
    return metrics.measure(env)


@gin.configurable
def run_stackelberg_simulation(env,
                               agent,
                               metrics,
                               num_steps,
                               seed=100,
                               agent_seed=100):
  """Performs a Stackelberg simulation.


  A Stackelberg Simulation involves a two player game between a Jury (Agent) and
  Contestants (Environment's population). In this setup the game proceeds as
  follows:
  1. Agent Publishes a classifier
  2. Contestants manipualte features to game the classifier
  3. Agent receives manipulated features and makes decision
  4. Environment receives agent's decision and calculates penalties/reward.

  In this case, we have folded steps 2, 3, 4 into the environment, where once
  the agent publishes its classifier, the feature manipulation, classification
  and reward calculation is done in one step in the environment.

  Args:
    env: A `core.FairnessEnv`.
    agent: A `core.Agent`.
    metrics: A list of `core.Metric` instances, a dict of {name: `core.Metric`}
      or a single `core.Metric` instance.
    num_steps: An integer indicating the numnber of steps to simulate.
    seed: An integer indicating a random seed.
    agent_seed: An integer indicating a random seed for the agent.

  Returns:
    A list of measurements if multiple metrics else a single measurement.
  """
  env.seed(seed)
  agent.seed(agent_seed)
  _ = env.reset()
  agent.action_space = env.action_space
  action = agent.initial_action()
  done = False
  print("Starting simulation")
  simulation_iterator = tqdm.trange if FLAGS.use_tqdm else range
  for _ in simulation_iterator(num_steps):
    # TODO(): Remove reward from this loop.
    observation, _, done, _ = env.step(action)
    # Update the agent with any changes to the observation or action space.
    agent.action_space, agent.observation_space = (env.action_space,
                                                   env.observation_space)
    action = agent.act(observation, done)
    if done:
      break

  print("Measuring metrics")
  if isinstance(metrics, list):
    return [metric.measure(env) for metric in metrics]
  elif isinstance(metrics, dict):
    return {name: metric.measure(env) for name, metric in metrics.items()}
  else:
    return metrics.measure(env)
