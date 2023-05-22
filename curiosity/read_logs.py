import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

rewards = ["scalar", "UCB", "visit_count"]
curiosity = [0.5, 1.5, 3]

#Function that loads the pickle files into a dictionary with reward+curiosity as key
def load_pickle_files(r, c):
    ""
    dict = {}
    with open(f"models/ppo_lending/400000_{c}_realfinal{r}/{r}_actions_callback.pkl", "rb") as f:
        dict[f"{r}_{c}"] = pickle.load(f).__dict__

    return dict

def sliding_average(lst, window_size):
    arr = np.array(lst)
    cum_sum = np.cumsum(arr)
    cum_sum[window_size:] = cum_sum[window_size:] - cum_sum[:-window_size]
    return cum_sum[window_size - 1:] / window_size

#Function that computes precision over time for a given reward and curiosity
def compute_precision_over_time(reward, curiosity, group):
    dict = load_pickle_files(reward, curiosity)
    actions = dict[f"{reward}_{curiosity}"][f'group{group}_actions']
    rewards = dict[f"{reward}_{curiosity}"][f'group{group}_rewards']
    TP = 0
    FP = 0
    precision = []
    for i in range(len(actions)):
        if (actions[i] == 1) & (rewards[i] > 0):
            TP += 1
        elif (actions[i] == 1) & (rewards[i] < 0):
            FP += 1
        if TP + FP == 0:
            pass
        else:
            precision.append(TP / (TP + FP))

    #Running average every 1000 steps
    precision = sliding_average(precision, 10000)
  
    return precision

#Function that computes recall over time for a given reward and curiosity
def compute_recall_over_time(reward, curiosity, group):
    dict = load_pickle_files(reward, curiosity)
    actions = dict[f"{reward}_{curiosity}"][f'group{group}_actions']
    rewards = dict[f"{reward}_{curiosity}"][f'group{group}_rewards']
    TP = 0
    FN = 0
    recall = []
    for i in range(len(actions)):
        if (actions[i] == 1) & (rewards[i] > 0):
            TP += 1
        elif (actions[i] == 0) & (rewards[i] > 0):
            FN += 1
        if TP + FN == 0:
            pass
        else:
            recall.append(TP / (TP + FN))
    #Running average every 1000 steps
    recall = sliding_average(recall, 10000)
    return recall

#Function that plots precision for given rewards and curiosities in one plot
def plot_precision(rewards, curiosities, group, save=False):
    plt.figure()
    for r in rewards:
        if r == 'scalar':
            color = 'red'
        elif r == 'UCB':
            color = 'blue'
        else:
            color = 'green'
        for g in group:
            if g == 0:
                marker = 'o'
            else:
                marker = 'x'
            for c in curiosities:
                if len(curiosities) > 1:
                    if c == 0.5:
                        color = 'red'
                    elif c == 1.5:
                        color = 'blue'
                    else:
                        color = 'green'
                precision = compute_precision_over_time(r, c, g)
                plt.plot(precision, label=f"{r}_{c}_group{g}", c=color, marker=marker, ms=10, markevery=25000)
    plt.legend()
    plt.grid()
    plt.title(f"Precision over time")
    plt.xlabel("Time")
    plt.ylabel("Precision")
    if save:
        if not os.path.exists("plots/precision"):
            os.makedirs("plots/precision")
        plt.savefig(f"plots/precision/precision_over_time_{rewards}_{curiosities}.png")
    plt.show()

#Function that plots recall for given rewards and curiosities in one plot
def plot_recall(rewards, curiosities, group, save=False):
    plt.figure()
    for r in rewards:
        if r == 'scalar':
            color = 'red'
        elif r == 'UCB':
            color = 'blue'
        else:
            color = 'green'
        for g in group:
            if g == 0:
                marker = 'o'
            else:
                marker = 'x'
            for c in curiosities:
                if len(curiosities) > 1:
                    if c == 0.5:
                        color = 'red'
                    elif c == 1.5:
                        color = 'blue'
                    else:
                        color = 'green'
                recall = compute_recall_over_time(r, c, g)
                plt.plot(recall, label=f"{r}_{c}_group{g}", c=color, marker=marker, ms=10, markevery=25000)
    plt.legend()
    plt.grid()
    plt.title(f"Recall over time")
    plt.xlabel("Time")
    plt.ylabel("Recall")
    if save:
        if not os.path.exists("plots/recall"):
            os.makedirs("plots/recall")
        plt.savefig(f"plots/recall/recall_over_time_{rewards}_{curiosities}.png")
    plt.show()

#Funciton that returns a list with a running average over the number of loans given for a given reward and curiosity
def compute_loans_over_time(reward, curiosity, group):
    dict = load_pickle_files(reward, curiosity)
    actions = dict[f"{reward}_{curiosity}"][f'group{group}_actions']
    #Running average every 1000 steps
    actions = sliding_average(actions, 10000)
    return actions

#Function that plots loans and precsion for given reward and curiosity in a single plot using 2 y-axes
def plot_precision_and_loans(reward, curiosity, group, save=False):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    color = 'red'
    marker = 'o'
    precision = compute_precision_over_time(reward, curiosity, group)
    loans = compute_loans_over_time(reward, curiosity, group)
    ax1.plot(precision, label=f"{reward}_{curiosity}_group{group}", c=color, marker=marker, ms=10, markevery=25000, alpha=0.6)
    ax2.plot(loans, label=f"{reward}_{curiosity}_group{group}_loans", c='blue', marker=marker, ms=10, markevery=25000, alpha=0.6)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.grid()
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Precision")
    ax2.set_ylabel("Loans")
    ax1.set_title(f"Precision and loans over time")
    if save:
        if not os.path.exists("plots/precision_and_loans"):
            os.makedirs("plots/precision_and_loans")
        plt.savefig(f"plots/precision_and_loans/precision_and_loans_over_time_{reward}_{curiosity}.png")
    plt.show()

if __name__ == "__main__":
    rewards = ['UCB']
    curiosities = [0.5, 1.5, 3]
    # plot_precision(rewards, curiosities, [0,1], save=True)
    # plot_recall(rewards, curiosities, [0,1], save=True)
    plot_precision_and_loans('UCB', 3, 1, save=False)