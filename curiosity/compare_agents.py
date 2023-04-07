# #Add parent folder
import sys
sys.path.append('../ml-fairness-gym')

#Lending environment without max bank cash
# from environments import lending
#Lending environment with max bank cash
from environments import lending_max3x as lending

from environments import lending_params

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from rl_agent import run_all

"""File to compare different agents on the lending environment."""

def grouped_barplot(data, group_names, title, xlabel, ylabel, x_tick_labels=None):
    """
    Creates a grouped barplot of n number of columns each with 2 groups using matplotlib.
    data: list of lists, where each inner list contains two values that should belong to one column in the plot
    group_names: list of strings, containing the names of the two groups in each column
    title: string, containing the title of the plot
    xlabel: string, containing the label for the x-axis
    ylabel: string, containing the label for the y-axis
    x_tick_labels: list of strings, containing the labels for each column
    """
    
    # Set the bar width
    bar_width = 0.35
    
    # Set the position of the bars on the x-axis
    x_pos = np.arange(len(data))
    
    # Create the plot object
    fig, ax = plt.subplots()
    
    # Loop through each column and plot the bars
    for i in range(len(data)):
        if i == 0:
            ax.bar(x_pos[i], data[i][0], bar_width, label=group_names[0], color='orange')
            ax.bar(x_pos[i] + bar_width, data[i][1], bar_width, label=group_names[1], color='blue')
        else:
            ax.bar(x_pos[i], data[i][0], bar_width, color='orange')
            ax.bar(x_pos[i] + bar_width, data[i][1], bar_width, color='blue')
    
    # Add labels, title, and legend
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x_pos)
    if x_tick_labels:
        ax.set_xticklabels(x_tick_labels)
    else:
        ax.set_xticklabels(range(1, len(data) + 1))
    ax.legend()
    
    plt.savefig(PLOT_PATH+'model_comparison.png', bbox_inches='tight')
    # Show the plot
    plt.show()


def create_boxplot(data, labels, title, x_label, y_label):
    """
    Creates boxplot using the provided data, labels, title, x_label and y_label.
    """
    # Set the figure size
    plt.figure(figsize=(10, 6))

    # Create the boxplot
    plt.boxplot(data, labels=labels)

    # Set the title and axis labels
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig(PLOT_PATH+'bank_cash_comparison.png', bbox_inches='tight')

    # Show the plot
    plt.show()


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

MODELS_TO_TRAIN = ['visit_count', 'scalar', 'UCB']
LEARNING_STEPS = 50000
PLOT_PATH = 'plots/'
test_results_dict = {}

for model in tqdm(MODELS_TO_TRAIN):
    test_results, baseline_results, action_callback = run_all(env_params, learning_steps=LEARNING_STEPS, 
                                                              rewards=model, show_plot=False, verbose=0,
                                                              learning_rate=0.0001,
                                                              n_steps=512, n_test_steps=1000)
    test_results_dict[model] = test_results
    test_results_dict[model+'actions'] = action_callback


#Create grouped barplot that shows the positive actions taken in the training phase for each model, 
# stratified by group
grouped_actions = [[np.mean(test_results_dict[model+'actions'].group0_actions), 
  np.mean(test_results_dict[model+'actions'].group1_actions)] for model in MODELS_TO_TRAIN]


grouped_barplot(grouped_actions, ['Group 0', 'Group 1'],
                 'Mean positive actions per group and model', 'Model',
                   'Mean positive actions', MODELS_TO_TRAIN)


#Create boxplot that shows the bank cash at the end of the test phase for each model
create_boxplot([test_results_dict[model]['bank_cash'] for model in MODELS_TO_TRAIN],
                            MODELS_TO_TRAIN, 'Bank cash at the end of the test phase',
                            'Model', 'Bank cash')
