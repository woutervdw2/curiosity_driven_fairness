# #Add parent folder
import sys
sys.path.append("/home/woutervdw2/Documents/thesis/code/ml-fairness-gym")

#Lending environment without max bank cash
# from environments import lending
#Lending environment with max bank cash
from environments import lending_max3x as lending

from environments import lending_params

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from rl_agent import run_all

"""File to compare different agents on the lending environment."""

def grouped_barplot(data, group_names, title, xlabel, ylabel, x_tick_labels=None, PLOT_PATH='plots/'):
    """
    Creates a grouped barplot of n number of columns each with 2 groups using matplotlib.
    data: list of lists, where each inner list contains two values that should belong to one column in the plot
    group_names: list of strings, containing the names of the two groups in each column
    title: string, containing the title of the plot
    xlabel: string, containing the label for the x-axis
    ylabel: string, containing the label for the y-axis
    x_tick_labels: list of strings, containing the labels for each column
    """
    
    #Check path exists, else create path
    if not os.path.exists(PLOT_PATH):
        os.makedirs(PLOT_PATH)
        
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


def create_boxplot(data, labels, title, x_label, y_label, PLOT_PATH='plots/'):
    """
    Creates boxplot using the provided data, labels, title, x_label and y_label.
    """

    #Check path exists, else create path
    if not os.path.exists(PLOT_PATH):
        os.makedirs(PLOT_PATH)

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


def compare_agents(env_params, models, **kwargs):
    """Compare the performance of different agents on the lending environment.
    Args:
        env_params: Parameters for the lending environment.
        models: List of models to compare.
        n_test_steps: Number of steps to run the test phase for.
        train: Whether to train the agents before testing them.
        show_plot: Whether to show the plot.
        kwargs: Additional arguments to pass to the run_all function.
    Returns:
        A list of the test results for each model.
    """

    PLOT_PATH = 'plots/'
    if 'model_name' in kwargs['kwargs']:
        PLOT_PATH = 'plots/'+kwargs['kwargs']['model_name']+'/'   

    #Check path exists, else create path
    if not os.path.exists(PLOT_PATH):
        os.makedirs(PLOT_PATH) 

    test_results_dict = {}
    for model in models:
        test_result, _, action_callback = run_all(env_params, rewards=model, **kwargs['kwargs'])
        test_results_dict[model] = test_result
        if action_callback is not None:
            test_results_dict[model+'actions'] = action_callback

    create_boxplot([test_results_dict[model]['bank_cash'] for model in models],
                       models,
                       'Bank cash at the end of the test phase',
                       'Model', 'Bank cash', PLOT_PATH=PLOT_PATH)
    
    #Create grouped barplot that shows the positive actions taken in the training phase for each model, 
    # stratified by group
    if action_callback is not None:
        grouped_actions = [[np.mean(test_results_dict[model+'actions'].group0_actions), 
        np.mean(test_results_dict[model+'actions'].group1_actions)] for model in MODELS_TO_TRAIN]


        grouped_barplot(grouped_actions, ['Group 0', 'Group 1'],
                        'Mean positive actions per group and model', 'Model',
                        'Mean positive actions', MODELS_TO_TRAIN, PLOT_PATH=PLOT_PATH)

    return 

if __name__ == '__main__':
    #Init environment
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

    # Set the number of steps to run the test phase for
    N_TEST_STEPS = 1000

    # Set the models to compare
    MODELS_TO_TRAIN = ['visit_count', 'scalar', 'UCB']

    # Compare the models
    compare_agents(env_params, MODELS_TO_TRAIN, N_TEST_STEPS, train=False, show_plot=False)