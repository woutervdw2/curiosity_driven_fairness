
"""Main file to run lending experiments expanded with curiosity agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# #Add parent folder
import sys
sys.path.append('../ml-fairness-gym')

import os
from absl import app
from absl import flags
from agents import threshold_policies
from experiments import lending_new as lending
from experiments import lending_plots
import matplotlib.pyplot as plt
import numpy as np
import simplejson as json

#AGENTS OPTIONS:
#Logistic, oracle
AGENT_NAME = 'rl'
if not os.path.exists(f'lending_{AGENT_NAME}_plots'):
   os.mkdir(f'lending_{AGENT_NAME}_plots')

flags.DEFINE_string('outfile', f'outfiles_{AGENT_NAME}.txt', 'Path to write out results.')
flags.DEFINE_string('plots_directory', f'lending_{AGENT_NAME}_plots', 'Directory to write out plots.')
flags.DEFINE_bool('equalize_opportunity', False,
                  'If true, apply equality of opportunity constraints.')
flags.DEFINE_integer('num_steps', 5,
                     'Number of steps to run the simulation.')

FLAGS = flags.FLAGS

# Control float precision in json encoding.
json.encoder.FLOAT_REPR = lambda o: repr(round(o, 3))

MAXIMIZE_REWARD = threshold_policies.ThresholdPolicy.MAXIMIZE_REWARD
EQUALIZE_OPPORTUNITY = threshold_policies.ThresholdPolicy.EQUALIZE_OPPORTUNITY


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  np.random.seed(100)
  group_0_prob = 0.5
  result = lending.Experiment(
      group_0_prob=group_0_prob,
      interest_rate=1.0,
      bank_starting_cash=10000,
      seed=200,
      num_steps=FLAGS.num_steps,
      burnin=200,
      cluster_shift_increment=0.01,
      include_cumulative_loans=True,
      return_json=False,
      threshold_policy=(EQUALIZE_OPPORTUNITY if FLAGS.equalize_opportunity else
                        MAXIMIZE_REWARD),
      agent=AGENT_NAME).run()

  title = ('Eq. opportunity' if FLAGS.equalize_opportunity else 'Max reward')
  metrics = result['metric_results']

  # Standalone figure of initial credit distribution
  fig = plt.figure(figsize=(4, 4))
  lending_plots.plot_credit_distribution(
      metrics['initial_credit_distribution'],
      'Initial',
      path=os.path.join(FLAGS.plots_directory,
                        'initial_credit_distribution.png')
      if FLAGS.plots_directory else None,
      include_median=True,
      figure=fig)

  # Initial and final credit distributions next to each other.
  fig = plt.figure(figsize=(8, 4))
  plt.subplot(1, 2, 1)
  lending_plots.plot_credit_distribution(
      metrics['initial_credit_distribution'],
      'Initial',
      path=None,
      include_median=True,
      figure=fig)
  plt.subplot(1, 2, 2)

  lending_plots.plot_credit_distribution(
      metrics['final_credit_distributions'],
      'Final - %s' % title,
      path=os.path.join(FLAGS.plots_directory, 'final_credit_distribution.png')
      if FLAGS.plots_directory else None,
      include_median=True,
      figure=fig)

  fig = plt.figure()
  lending_plots.plot_bars(
      metrics['recall'],
      title='Recall - %s' % title,
      path=os.path.join(FLAGS.plots_directory, 'recall.png')
      if FLAGS.plots_directory else None,
      figure=fig)

  fig = plt.figure()
  lending_plots.plot_bars(
      metrics['precision'],
      title='Precision - %s' % title,
      ylabel='Precision',
      path=os.path.join(FLAGS.plots_directory, 'precision.png')
      if FLAGS.plots_directory else None,
      figure=fig)

  fig = plt.figure()
  lending_plots.plot_cumulative_loans(
      {'demo - %s' % title: metrics['cumulative_loans']},
      path=os.path.join(FLAGS.plots_directory, 'cumulative_loans.png')
      if FLAGS.plots_directory else None,
      figure=fig)

  print('Profit %s %f' % (title, result['metric_results']['profit rate']))
  plt.show()

  if FLAGS.outfile:
    with open(FLAGS.outfile, 'w') as f:
        for key, value in result['metric_results'].items():
            f.write(f'{key}: {value} \n')


if __name__ == '__main__':
  app.run(main)
