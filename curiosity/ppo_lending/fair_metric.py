import os
import numpy as np
import pandas as pd
from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio
from tqdm import tqdm

#Function that reads txt file and collects a list of actions_group0 and actions_group1 
# and a list of defaults and returns them in a dictionary
def read_txt_file(r, c, dir_path):
    file = f"{dir_path}/models/ppo_lending/500000_{c}{r}/{r}_best/sim_results.txt"
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

def compute_dem_parity(reward, curiosity, dir_path):
    for r in tqdm(reward):
        for c in tqdm(curiosity):
            print("Computing demographic parity for reward: ", r, " and curiosity: ", c, "\n")
            dict = read_txt_file(r, c, dir_path)
            #Not because action should be 1 if the person is not defaulting
            y_true = dict[f"{r}_{c}"]['group0_default'] + dict[f"{r}_{c}"]['group1_default']
            y_true = [False if x else True for x in y_true]
            y_pred = dict[f"{r}_{c}"]['group0_actions'] + dict[f"{r}_{c}"]['group1_actions']
            sens_group = [0]*len(dict[f"{r}_{c}"]['group0_actions']) + [1]*len(dict[f"{r}_{c}"]['group1_actions'])
            difference = demographic_parity_difference(y_true, y_pred, sensitive_features=sens_group)
            ratio = demographic_parity_ratio(y_true, y_pred, sensitive_features=sens_group)
            #Write to file
            with open(f"{dir_path}/fairness/dem_parity.txt", "a") as f:
                f.write(f"{r}_{c}: differenc: {difference}, ratio: {ratio}\n")

def compute_cond_dem_parity(reward, curiosity, dir_path):
    save_dict = {}
    for r in tqdm(reward):
        for c in tqdm(curiosity):
            print("Computing conditional demographic parity for reward: ", r, " and curiosity: ", c, "\n")
            dict = read_txt_file(r, c, dir_path)
            y_true = dict[f"{r}_{c}"]['group0_default'] + dict[f"{r}_{c}"]['group1_default']
            y_true = [False if x else True for x in y_true]
            y_pred = dict[f"{r}_{c}"]['group0_actions'] + dict[f"{r}_{c}"]['group1_actions']
            group = [0]*len(dict[f"{r}_{c}"]['group0_actions']) + [1]*len(dict[f"{r}_{c}"]['group1_actions'])
            features = dict[f"{r}_{c}"]['group0_features'] + dict[f"{r}_{c}"]['group1_features']
            for y, pred, group, features in zip(y_true, y_pred, group, features):
                if pred == 1:
                    try:
                        save_dict[f"{r}_{c}_{group}_{features}_true"] += 1
                        save_dict[f"{r}_{c}_{group}_{features}_total"] += 1
                    except:
                        save_dict[f"{r}_{c}_{group}_{features}_true"] = 1
                        save_dict[f"{r}_{c}_{group}_{features}_total"] = 1
                else:
                    try:
                        save_dict[f"{r}_{c}_{group}_{features}_total"] += 1
                    except:
                        save_dict[f"{r}_{c}_{group}_{features}_total"] = 1
            for group in [0,1]:
                for features in range(0,6):
                    with open(f"{dir_path}/fairness/cond_dem_parity.txt", "a") as f:
                        try:
                            f.write(f"{r}_{c}_{group}_{features} total: {save_dict[f'{r}_{c}_{group}_{features}_total']}\n")
                            f.write(f"{r}_{c}_{group}_{features} true: {save_dict[f'{r}_{c}_{group}_{features}_true']}\n")
                        except:
                            f.write(f"{r}_{c}_{group}_{features} total: 0\n")
                            f.write(f"{r}_{c}_{group}_{features} true: 0\n")
        

if __name__ == "__main__":
    dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    reward = ["UCB", "scalar", "visit_count"]
    curiosity = [0.5, 1.5, 3, 6]
    compute_dem_parity(reward,curiosity,dir_path)
    compute_cond_dem_parity(reward,curiosity,dir_path)