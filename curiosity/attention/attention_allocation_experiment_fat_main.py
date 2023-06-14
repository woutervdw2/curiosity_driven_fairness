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

"""Main file to run attention allocation experiments.

Note this file can take a significant amount of time to run all experiments
since experiments are being repeated multiple times and the results averaged.
To run experiments fewer times, change the experiments.num_runs parameter to
10.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("/home/woutervdw2/Documents/thesis/code/ml-fairness-gym")

import json
import os

from absl import app
from absl import flags
from agents import allocation_agents
from agents import random_agents
from agents import rl_agents
from environments import attention_allocation_copy as attention_allocation
from experiments import attention_allocation_experiment_copy as attention_allocation_experiment
from experiments import attention_allocation_experiment_plotting
import numpy as np



def _get_base_env_params():
  return attention_allocation.Params(
      n_locations=6,
      prior_incident_counts=(500, 500, 500, 500, 500, 500),
      incident_rates=[8, 6, 4, 3, 1.5, 0.5],
      n_attention_units=4,
      miss_incident_prob=(0., 0., 0., 0., 0., 0.),
      extra_incident_prob=(0., 0., 0., 0., 0., 0.),
      dynamic_rate=0.0)


def _setup_experiment():
  return attention_allocation_experiment.Experiment(
      num_runs=1,
      num_steps=10000,
      num_workers=1,
      seed=10,
      env_class=attention_allocation.LocationAllocationEnv,
      env_params=_get_base_env_params())


def _print_discovered_missed_incidents_report(value, report):
  discovered_incidents = np.array(report['metrics']['discovered_incidents'])
  discovered_total = np.sum(discovered_incidents)
  missed_incidents = np.array(
      report['metrics']['occurred_incidents']) - np.array(
          report['metrics']['discovered_incidents'])
  missed_total = np.sum(missed_incidents)

  print(
      'REPORT dynamic_value: {}\ndiscovered_total: {}\nmissed_total: {}\ndiscovered_locations: {}\nmissed_locations: {}\n'
      .format(value, discovered_total, missed_total, discovered_incidents,
              missed_incidents))


def mle_greedy_alpha5_agent_resource_all_dynamics():
  """Run experiments on a greedy-epsilon mle agent, epsilon=0.1, across dynamics."""
  dynamic_values_to_test = [0.0, 0.01, 0.05, 0.1, 0.15]
  experiment = _setup_experiment()
  experiment.agent_class = allocation_agents.MLEGreedyAgent
  experiment.agent_params = allocation_agents.MLEGreedyAgentParams(
      burn_steps=25, window=100, alpha=0.75)

  reports_dict = {}

  for value in dynamic_values_to_test:
    print('Running an experiment...')
    experiment.env_params.dynamic_rate = value
    json_report = attention_allocation_experiment.run(experiment)
    report = json.loads(json_report)

    print('\n\nMLE Greedy Fair Agent, 6 attention units, alpha=0.75')
    _print_discovered_missed_incidents_report(value, report)
    output_filename = 'mle_greedy_fair_alpha75_6units_%f.json' % value
    with open(os.path.join(FLAGS.output_dir, output_filename), 'w') as f:
      json.dump(report, f)

    reports_dict[value] = json_report
  return reports_dict

def mle_greedy_agent_resource_all_dynamics():
  """Run experiments on a greedy-epsilon mle agent, epsilon=0.1, across dynamics."""
  dynamic_values_to_test = [0.0, 0.01, 0.05, 0.1, 0.15]
  experiment = _setup_experiment()
  experiment.agent_class = allocation_agents.MLEGreedyAgent
  experiment.agent_params = allocation_agents.MLEGreedyAgentParams(
      burn_steps=25, window=100)

  reports_dict = {}

  for value in dynamic_values_to_test:
    print('Running an experiment...')
    experiment.env_params.dynamic_rate = value
    json_report = attention_allocation_experiment.run(experiment)
    report = json.loads(json_report)

    print('\n\nMLE Greedy Agent, 6 attention units')
    _print_discovered_missed_incidents_report(value, report)
    output_filename = 'mle_greedy_6units_%f.json' % value
    with open(os.path.join(FLAGS.output_dir, output_filename), 'w') as f:
      json.dump(report, f)

    reports_dict[value] = json_report
  return reports_dict


def uniform_agent_resource_all_dynamics():
  """Run experiments on a uniform agent across dynamic rates."""

  dynamic_values_to_test = [0.0, 0.01, 0.05, 0.1, 0.15]
  experiment = _setup_experiment()
  experiment.agent_class = random_agents.RandomAgent

  reports_dict = {}

  for value in dynamic_values_to_test:
    print('Running an experiment...')
    experiment.env_params.dynamic_rate = value
    json_report = attention_allocation_experiment.run(experiment)
    report = json.loads(json_report)

    print('\n\nUniform Random Agent, 6 attention units')
    _print_discovered_missed_incidents_report(value, report)
    output_filename = 'uniform_6units_%f.json' % value
    with open(os.path.join(FLAGS.output_dir, output_filename), 'w') as f:
      json.dump(report, f)

    reports_dict[value] = json_report
  return reports_dict


def mle_agent_epsilon_1_resource_all_dynamics():
  """Run experiments on a greedy-epsilon mle agent, epsilon=0.1, across dynamics."""
  dynamic_values_to_test = [0.0, 0.01, 0.05, 0.1, 0.15]
  experiment = _setup_experiment()
  experiment.agent_class = allocation_agents.MLEProbabilityMatchingAgent
  experiment.agent_params = allocation_agents.MLEProbabilityMatchingAgentParams(
  )
  experiment.agent_params.burn_steps = 25
  experiment.agent_params.window = 100

  reports_dict = {}

  for value in dynamic_values_to_test:
    print('Running an experiment...')
    experiment.env_params.dynamic_rate = value
    json_report = attention_allocation_experiment.run(experiment)
    report = json.loads(json_report)

    print('\n\nMLE Agent, 6 attention units, epsilon=0.1')
    _print_discovered_missed_incidents_report(value, report)
    output_filename = 'mle_epsilon.1_6units_%f.json' % value
    with open(os.path.join(FLAGS.output_dir, output_filename), 'w') as f:
      json.dump(report, f)

    reports_dict[value] = json_report
  return reports_dict


def mle_agent_epsilon_5_resource_all_dynamics():
  """Run experiments on a greedy-epsilon mle agent, epsilon=0.6, across dynamics."""
  dynamic_values_to_test = [0.0, 0.01, 0.05, 0.1, 0.15]
  experiment = _setup_experiment()
  experiment.agent_class = allocation_agents.MLEProbabilityMatchingAgent
  experiment.agent_params = allocation_agents.MLEProbabilityMatchingAgentParams(
  )
  experiment.agent_params.burn_steps = 25
  experiment.agent_params.epsilon = 0.5
  experiment.agent_params.window = 100

  reports_dict = {}

  for value in dynamic_values_to_test:
    experiment.env_params.dynamic_rate = value
    json_report = attention_allocation_experiment.run(experiment)
    report = json.loads(json_report)

    print('\n\nMLE Agent, 6 attention units, epsilon=0.5')
    _print_discovered_missed_incidents_report(value, report)
    output_filename = 'mle_epsilon.5_6units_%f.json' % value
    with open(os.path.join(FLAGS.output_dir, output_filename), 'w') as f:
      json.dump(report, f)

    reports_dict[value] = json_report
  return reports_dict


FLAGS = flags.FLAGS
flags.DEFINE_string('output_dir', './results_attention_6',
                    'Output directory to write results to.')

def rl_agent_all_dynamics(model_name):
  dynamic_values_to_test = [0.0, 0.01, 0.05, 0.1, 0.15]
  experiment = _setup_experiment()
  experiment.agent_class = rl_agents.RlAgent
  experiment.model_name = model_name
  
  reports_dict = {}

  for value in dynamic_values_to_test:
    print('Running an experiment...')
    experiment.env_params.dynamic_rate = value
    json_report = attention_allocation_experiment.run(experiment)
    report = json.loads(json_report)

    print('\n\n RL agent, 4 attention units')
    _print_discovered_missed_incidents_report(value, report)
    output_filename = 'rl_agent_4units_%f.json' % value
    if not os.path.exists(FLAGS.output_dir):
      os.makedirs(FLAGS.output_dir)
    with open(os.path.join(FLAGS.output_dir, output_filename), 'w') as f:
      json.dump(report, f)

    reports_dict[value] = json_report
  return reports_dict

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # greedy_fair_reports = mle_greedy_alpha5_agent_resource_all_dynamics()
  # greedy_reports = mle_greedy_agent_resource_all_dynamics()
  # uniform_reports = uniform_agent_resource_all_dynamics()
  # mle1_reports = mle_agent_epsilon_1_resource_all_dynamics()
  # mle5_reports = mle_agent_epsilon_5_resource_all_dynamics()
  # scalar_15 = rl_agent_all_dynamics("scalar_1.5/scalar_best/")
  # ucb_15 = rl_agent_all_dynamics("UCB_1.5/UCB_best/")
  # visit_count_15 = rl_agent_all_dynamics("visit_count_1.5/visit_count_best/")
  
  scalar_3 = rl_agent_all_dynamics("scalar_3/scalar_best/")
  ucb_3 = rl_agent_all_dynamics("UCB_3/UCB_best/")
  visit_count_3 = rl_agent_all_dynamics("visit_count_3/visit_count_best/")
  
  scalar_6 = rl_agent_all_dynamics("scalar_6/scalar_best/")
  ucb_6 = rl_agent_all_dynamics("UCB_6/UCB_best/")
  visit_count_6 = rl_agent_all_dynamics("visit_count_6/visit_count_best/")
  
  
  agent_names15 = ["scalar_1.5", "UCB_1.5", "visit_count_1.5"]
  agent_names3 = ["scalar_3", "UCB_3", "visit_count_3"]
  agent_names6 = ["scalar_6", "UCB_6", "visit_count_6"]
  
  agent_names_ucb = ["UCB_1.5", "UCB_3", "UCB_6"]
  agent_names_scalar = ["scalar_1.5", "scalar_3", "scalar_6"]
  agent_names_visit_count = ["visit_count_1.5", "visit_count_3", "visit_count_6"]
  
  # dataframe = attention_allocation_experiment_plotting.create_dataframe_from_results(
  #     agent_names15, [
  #         scalar_15, ucb_15, visit_count_15
  #     ])
  # loc_dataframe = attention_allocation_experiment_plotting.create_dataframe_from_results(
    #   agent_names15, [
    #       scalar_15, ucb_15, visit_count_15
    #   ],
    #   separate_locations=True)
  
  # dataframe = attention_allocation_experiment_plotting.create_dataframe_from_results(
  #     agent_names3, [
  #         scalar_3, ucb_3, visit_count_3
  #     ])
  
  # loc_dataframe = attention_allocation_experiment_plotting.create_dataframe_from_results(
  #     agent_names3, [
  #         scalar_3, ucb_3, visit_count_3
  #     ],  
  #     separate_locations=True)
  
  dataframe = attention_allocation_experiment_plotting.create_dataframe_from_results(
      agent_names6, [
          scalar_6, ucb_6, visit_count_6
      ])
  
  loc_dataframe = attention_allocation_experiment_plotting.create_dataframe_from_results(
      agent_names6, [
          scalar_6, ucb_6, visit_count_6
      ],
      separate_locations=True)
  
  # dataframe = attention_allocation_experiment_plotting.create_dataframe_from_results(
  #     agent_names_ucb, [
  #         ucb_15, ucb_3, ucb_6
  #     ])
  
  # loc_dataframe = attention_allocation_experiment_plotting.create_dataframe_from_results(
  #     agent_names_ucb, [
  #         ucb_15, ucb_3, ucb_6
  #     ],  
  #     separate_locations=True)
  
  
  attention_allocation_experiment_plotting.plot_discovered_missed_clusters(
      loc_dataframe,
      os.path.join(FLAGS.output_dir, 'dynamic_rate_across_agents_locations'))
  attention_allocation_experiment_plotting.plot_total_miss_discovered(
      dataframe, os.path.join(FLAGS.output_dir, 'dynamic_rate_across_agents'))
  attention_allocation_experiment_plotting.plot_discovered_occurred_ratio_locations(
      loc_dataframe,
      os.path.join(FLAGS.output_dir, 'discovered_to_occurred_locations'))
  attention_allocation_experiment_plotting.plot_discovered_occurred_ratio_range(
      dataframe, os.path.join(FLAGS.output_dir, 'discovered_to_occurred_range'))
  
  attention_allocation_experiment_plotting.plot_occurence_action_single_dynamic(
      json.loads(scalar_6[0.0]),
      os.path.join(FLAGS.output_dir,
                    'rl_agent_incidents_actions_over_time'))


if __name__ == '__main__':
  app.run(main)