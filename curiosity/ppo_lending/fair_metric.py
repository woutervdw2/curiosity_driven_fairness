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
                group_total = 0
                len_group = len(dict[f"{r}_{c}"][f'group{group}_actions'])
                for features in range(0,6):
                    with open(f"{dir_path}/fairness/cond_dem_parity.txt", "a") as f:
                        try:
                            true = save_dict[f'{r}_{c}_{group}_{features}_true']
                            total = save_dict[f'{r}_{c}_{group}_{features}_total']
                            group_total += (true/total)*(total/len_group)
                            f.write(f"{r}_{c}_{group}_{features}: {true/total}\n")
                        except:
                            f.write(f"{r}_{c}_{group}_{features} total: 0\n")
                            f.write(f"{r}_{c}_{group}_{features} true: 0\n")
                with open(f"{dir_path}/fairness/cond_dem_parity.txt", "a") as f:
                    f.write(f"{r}_{c}_{group}_total: {group_total}\n")
                    
def pred_parity(reward, curiosity, dir_path):

    for r in tqdm(reward):
        for c in tqdm(curiosity):
            print("Computing predictive parity for reward: ", r, " and curiosity: ", c, "\n")
            dict = read_txt_file(r, c, dir_path)
            y_true = dict[f"{r}_{c}"]['group0_default'] + dict[f"{r}_{c}"]['group1_default']
            y_true = [False if x else True for x in y_true]
            y_pred = dict[f"{r}_{c}"]['group0_actions'] + dict[f"{r}_{c}"]['group1_actions']
            group = [0]*len(dict[f"{r}_{c}"]['group0_actions']) + [1]*len(dict[f"{r}_{c}"]['group1_actions'])
            
            #Compute predictive parity ppv
            p_y1_d1_g0 = np.where((np.array(y_true) == 1) & (np.array(y_pred) == 1) & (np.array(group) == 0))[0].shape[0]
            ppv_y1_d1_go = p_y1_d1_g0/np.where((np.array(y_true) == 1) & (np.array(group) == 0))[0].shape[0]
            p_y1_d1_g1 = np.where((np.array(y_true) == 1) & (np.array(y_pred) == 1) & (np.array(group) == 1))[0].shape[0]
            ppv_y1_d1_g1 = p_y1_d1_g1/np.where((np.array(y_true) == 1) & (np.array(group) == 1))[0].shape[0]
            ratio_ppv = ppv_y1_d1_g1/ppv_y1_d1_go
            
            #Compute false positive errore rate
            p_y0_d1_g0 = np.where((np.array(y_true) == 0) & (np.array(y_pred) == 1) & (np.array(group) == 0))[0].shape[0]
            FPR_y0_d1_g0 = p_y0_d1_g0/np.where((np.array(y_true) == 0) & (np.array(group) == 0))[0].shape[0]
            p_y0_d1_g1 = np.where((np.array(y_true) == 0) & (np.array(y_pred) == 1) & (np.array(group) == 1))[0].shape[0]
            FPR_y0_d1_g1 = p_y0_d1_g1/np.where((np.array(y_true) == 0) & (np.array(group) == 1))[0].shape[0]
            ratio_fpr = FPR_y0_d1_g1/FPR_y0_d1_g0
            
            #Compute false negative error rate
            p_y1_d0_g0 = np.where((np.array(y_true) == 1) & (np.array(y_pred) == 0) & (np.array(group) == 0))[0].shape[0]
            FNR_y1_d0_g0 = p_y1_d0_g0/np.where((np.array(y_true) == 1) & (np.array(group) == 0))[0].shape[0]
            p_y1_d0_g1 = np.where((np.array(y_true) == 1) & (np.array(y_pred) == 0) & (np.array(group) == 1))[0].shape[0]
            FNR_y1_d0_g1 = p_y1_d0_g1/np.where((np.array(y_true) == 1) & (np.array(group) == 1))[0].shape[0]
            ratio_fnr = FNR_y1_d0_g1/FNR_y1_d0_g0
            
            
            with open(f"{dir_path}/fairness/pred_parity.txt", "a") as f:
                f.write(f"{r}_{c}_ppv_y1_d1_g0: {ppv_y1_d1_go}\n")
                f.write(f"{r}_{c}_ppv_y1_d1_g1: {ppv_y1_d1_g1}\n")
                f.write(f"{r}_{c}_ratio_ppv: {ratio_ppv}\n")
                f.write(f"{r}_{c}_FPR_y0_d1_g0: {FPR_y0_d1_g0}\n")
                f.write(f"{r}_{c}_FPR_y0_d1_g1: {FPR_y0_d1_g1}\n")
                f.write(f"{r}_{c}_ratio_fpr: {ratio_fpr}\n")
                f.write(f"{r}_{c}_FNR_y1_d0_g0: {FNR_y1_d0_g0}\n")
                f.write(f"{r}_{c}_FNR_y1_d0_g1: {FNR_y1_d0_g1}\n")
                f.write(f"{r}_{c}_ratio_fnr: {ratio_fnr}\n")

if __name__ == "__main__":
    dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    reward = ["UCB", "scalar", "visit_count"]
    curiosity = [0.5, 1.5, 3, 6]
    compute_dem_parity(reward,  curiosity,  dir_path)
    compute_cond_dem_parity(reward, curiosity,  dir_path)
    pred_parity(reward, curiosity,  dir_path)