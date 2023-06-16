import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import sys

#Function that loads the pickle files into a dictionary with reward+curiosity as key
def load_pickle_files(r, c, dir_path):
    ""
    dict = {}
    with open(f"{dir_path}models/ppo_lending/1000000_{c}{r}/{r}_actions_callback.pkl", "rb") as f:
        dict[f"{r}_{c}"] = pickle.load(f).__dict__

    return dict

#Function that reads txt file and collects a list of actions_group0 and actions_group1 
# and a list of defaults and returns them in a dictionary
def read_txt_file(r, c, dir_path):
    file = f"{dir_path}models/ppo_lending/500000_{c}{r}/{r}_best/sim_results.txt"
    dict = {}
    with open(file, "r") as f:
        lines = f.readlines()
        lines = [line for line in lines if len(line)>100]
        actions_group0 = eval(lines[0])
        actions_group1 = eval(lines[1])
        default0 = eval(lines[2])
        default1 = eval(lines[3])
        features0 = eval(lines[4])
        features1 = eval(lines[5])
    dict[f"{r}_{c}"] = {'group0_actions': actions_group0, 'group1_actions': actions_group1,
                             'group0_default': default0, 'group1_default': default1,
                             'group0_features': features0, 'group1_features': features1} 

    return dict



def sliding_average(lst, window_size):
    arr = np.array(lst)
    cum_sum = np.cumsum(arr)
    cum_sum[window_size:] = cum_sum[window_size:] - cum_sum[:-window_size]
    return cum_sum[window_size - 1:] / window_size

#Function that computes precision over time for a given reward and curiosity
def compute_precision_over_time(reward, curiosity, group, sim=False, dir_path=None):
    if sim:
        dict = read_txt_file(reward, curiosity, dir_path)
    else:
        dict = load_pickle_files(reward, curiosity, dir_path)
    actions = dict[f"{reward}_{curiosity}"][f'group{group}_actions']
    rewards = dict[f"{reward}_{curiosity}"][f'group{group}_default']
    TP = 0
    FP = 0
    precision = []
    for i in range(len(actions)):
        if (actions[i] == 1) & (rewards[i] == False):
            TP += 1
        elif (actions[i] == 1) & (rewards[i] == True):
            FP += 1
        if TP + FP == 0:
            precision.append(1)
        else:
            precision.append(TP / (TP + FP))

    #Running average every 1000 steps
    precision = sliding_average(precision, 50)
  
    return precision

#Function that computes recall over time for a given reward and curiosity
def compute_recall_over_time(reward, curiosity, group, sim=False, dir_path=None):
    if sim:
        dict = read_txt_file(reward, curiosity, dir_path)
    else:
        dict = load_pickle_files(reward, curiosity, dir_path)

    actions = dict[f"{reward}_{curiosity}"][f'group{group}_actions']
    rewards = dict[f"{reward}_{curiosity}"][f'group{group}_default']
    TP = 0
    FN = 0
    recall = []
    for i in range(len(actions)):
        if (actions[i] == 1) & (rewards[i] == False):
            TP += 1
        elif (actions[i] == 0) & (rewards[i] == False):
            FN += 1
        if TP + FN == 0:
            recall.append(1)
        else:
            recall.append(TP / (TP + FN))
    #Running average every 1000 steps
    recall = sliding_average(recall, 50)
    return recall

#Function that plots precision for given rewards and curiosities in one plot
def plot_precision(rewards, curiosities, group, save=False, sim=False, dir_path=None):
    plt.figure(figsize=(10, 6))
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
                    elif c == 3.0:
                        color = 'green'
                    else:
                        color = 'orange'
                precision = compute_precision_over_time(r, c, g, sim=sim, dir_path=dir_path)
                plt.plot(precision, label=f"{r}_{c}_group{g+1}", c=color, marker=marker, ms=10, markevery=1000)
    plt.legend()
    plt.grid()
    plt.title(f"Precision over time")
    plt.xlabel("Time")
    plt.ylabel("Precision")
    if save:
        if not os.path.exists(dir_path+"final_plots/precision"):
            os.makedirs(dir_path+"final_plots/precision")
        plt.savefig(f"{dir_path}final_plots/precision/precision_over_time_{rewards}_{curiosities}_{group}.png", bbox_inches='tight')
    plt.show()

#Function that plots recall for given rewards and curiosities in one plot
def plot_recall(rewards, curiosities, group, save=False, sim=False, dir_path=None):
    plt.figure(figsize=(11, 6))
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
                    elif c == 3.0:
                        color = 'green'
                    else:
                        color = 'orange'
                recall = compute_recall_over_time(r, c, g, sim=sim, dir_path=dir_path)
                plt.plot(recall, label=f"{r}_{c}_group{g+1}", c=color, marker=marker, ms=10, markevery=1000,
                         alpha=.8)
    plt.legend()
    plt.grid()
    plt.title(f"Recall over time")
    plt.xlabel("Time")
    plt.ylabel("Recall")
    if save:
        if not os.path.exists(dir_path+"final_plots/recall"):
            os.makedirs(dir_path+"final_plots/recall")
        plt.savefig(f"{dir_path}final_plots/recall/recall_over_time_{rewards}_{curiosities}_{group}.png")
    plt.show()

#Funciton that returns a list with a running average over the number of loans given for a given reward and curiosity
def compute_loans_over_time(reward, curiosity, group, sim=False, dir_path=None):
    if sim:
        dict = read_txt_file(reward, curiosity, dir_path)
    else:
        dict = load_pickle_files(reward, curiosity, dir_path)
    actions = dict[f"{reward}_{curiosity}"][f'group{group}_actions']
    #Running average every 1000 steps
    actions = sliding_average(actions, 50)
    return actions

def plot_loans(rewards, curiosities, group, save=False, sim=False, dir_path=None):
    plt.figure()
    color_changed = False
    for c in curiosities:
        if len(curiosities) > 1:
            if c == 0.5:
                color = 'red'
            elif c == 1.5:
                color = 'blue'
            elif c == 3.0:
                color = 'green'
            else:
                color = 'orange'
            color_changed = False
        for r in rewards:
            if color_changed:
                pass
            elif r == 'scalar':
                color = 'red'
            elif r == 'UCB':
                color = 'blue'
            else:
                color = 'green'
            for g in group:
                if g == 0:
                    marker = 'o'
                    if not color_changed:
                        color = 'red'
                else:
                    marker = 'x'
                loans = compute_loans_over_time(r, c, g, sim=sim, dir_path=dir_path)
                plt.plot(loans, label=f"{r}_{c}_group{g+1}", ms=10, markevery=1000, c=color, marker=marker, alpha=.8)
    plt.legend()
    plt.grid()
    plt.title(f"Loans over time")
    plt.xlabel("Time")
    plt.ylabel("Loans")
    if save:
        if not os.path.exists(dir_path+"final_plots/loans"):
            os.makedirs(dir_path+"final_plots/loans")
        plt.savefig(f"{dir_path}final_plots/loans/loans_over_time_{rewards}_{curiosities}.png", bbox_inches='tight')
    plt.show()
    
#Function that plots loans and precsion for given reward and curiosity in a single plot using 2 y-axes
def plot_precision_and_loans(reward, curiosity, group, save=False, sim=False, dir_path=None):
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()
    color = 'red'
    marker = 'o'
    precision = compute_precision_over_time(reward, curiosity, group, sim=sim, dir_path=dir_path)
    loans = compute_loans_over_time(reward, curiosity, group, sim=sim, dir_path=dir_path)
    ax1.plot(precision, label=f"{reward}-c={curiosity} precision", c=color, marker=marker, ms=10, markevery=1000, alpha=0.6)
    ax2.plot(loans, label=f"{reward}-c={curiosity} loans", c='blue', marker=marker, ms=10, markevery=1000, alpha=0.6)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.grid()
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Precision")
    ax2.set_ylabel("Loans")
    ax1.set_title(f"Precision and loans over time")
    if save:
        if not os.path.exists(dir_path+"final_plots/precision_and_loans"):
            os.makedirs(dir_path+"final_plots/precision_and_loans")
        plt.savefig(f"{dir_path}final_plots/precision_and_loans/precision_and_loans_over_time_{reward}_{curiosity}_{group}.png", bbox_inches='tight')
    plt.show()

def barplot_precision(reward, curiosity, group, save=False, sim=False, dir_path=None):
    dict = {}
    for r in reward:
        for c in curiosity:
            for g in group:
                dict[f'{r}_{c}_{g}'] = compute_precision_over_time(r, c, g, sim=sim, dir_path=dir_path)[-1]
    plt.figure()
    plt.bar(dict.keys(), dict.values(), color=['red', 'blue', 'green'], alpha=.8)
    plt.title(f"Precision comparison with c=0.5")
    plt.xlabel("Reward and curiosity")
    plt.ylabel("Precision")
    plt.xticks(rotation=45, labels=[f'Scalar', f'UCB', f'Visit-Counts'], ticks=[0, 1, 2])
    plt.tight_layout()
    plt.savefig(f"{dir_path}final_plots/precision_comparison_{r}_{c}_{g}.png")
    plt.show()
                
                
                
if __name__ == "__main__":
    dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))+"/"
    rewards = ['visit_count', "UCB", "scalar"]
    curiosities = [6]
    groups = [1]
    # plot_precision(rewards, curiosities, groups, save=True, sim=True, dir_path=dir_path)
    # barplot_precision(rewards, curiosities, groups, save=True, sim=True, dir_path=dir_path) 
    # plot_recall(rewards, curiosities, groups, save=True, sim=True, dir_path=dir_path)
    # plot_precision_and_loans('scalar', 3, 1, save=True, sim=True, dir_path=dir_path)
    # plot_precision_and_loans('scalar', 1.5, 1, save=True, sim=True, dir_path=dir_path)
    plot_loans(rewards, curiosities, groups, save=True, sim=True, dir_path=dir_path)
    
    