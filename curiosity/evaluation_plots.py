import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

#Function that goes over a directory and 
# in directories containing a certain string will search for files with a given extension
#Returns a dicionary with the name of the directory as key and a list of files as value
def search_files(directory, search_string, extension):
    result = {}
    
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            if search_string in dir_name:
                dir_path = os.path.join(root, dir_name)

                #Find string after final in dir_name
                if "boosted" in dir_name:
                    index = dir_name.find("boosted")
                else:
                    index = dir_name.find("final")
                    lentgh = len("final")
                    if index == -1:
                        pass
                    else:
                        reward = dir_name[index + lentgh:].strip()+"_best"
                    
                    dir_path = os.path.join(dir_path, reward)

                
                for file_name in os.listdir(dir_path):
                    if file_name.endswith(extension):
                        file_names = file_name
                        break
                
                if file_names:
                    result[dir_name] = file_names
    
    return result



final_files = search_files('models/ppo_lending', 'final', '.txt')

#Function that goes over a dictionary of {path: file}, finds in each txt file the line with the given string
# and returns a dictionary with the name of the directory as key and the value is a list of the values
# in the line containing the string
# def search_lines(dictionary, search_string, path="models/ppo_lending/"):
#     result = {}
    
#     for file_path, file in dictionary.items():
#         total_path = path+file_path+"/"+file
#         if file.endswith(".txt"):
#             with open(total_path, "r") as f:
#                 lines = f.readlines()
#                 values = []
                
#                 for line in lines:
#                     if search_string in line:
#                         line_values = line[len(search_string):].strip().split()
#                         values.extend(line_values)
                
#                 if values:
#                     directory_name = os.path.dirname(total_path)
#                     result[directory_name] = values
    
#     return result

import re

def parse_text_files(dictionary, root="models/ppo_lending"):
    result = {}
    
    for path, file in dictionary.items():
        total_path = f"{root}/{path}/{file}"
        if total_path.endswith(".txt"):
            with open(total_path, "r") as f:
                content = f.read()
                
                initial_credit_dist_match = re.search(r"initial_credit_distribution: (.+)", content)
                final_credit_dist_match = re.search(r"final_credit_distributions: (.+)", content)
                recall_match = re.search(r"recall: (.+)", content)
                precision_match = re.search(r"precision: (.+)", content)
                profit_rate_match = re.search(r"profit rate: (.+)", content)
                accuracy_match = re.search(r"accuracy: (.+)", content)
                recall_features_match = re.search(r"recall_features: (.+)", content)
                precision_features_match = re.search(r"precision_features: (.+)", content)
                cumulative_loans_match = re.search(r"cumulative_loans: (.+)", content)
                cumulative_recall_match = re.search(r"cumulative_recall: (.+)", content)
                
                parsed_data = {}
                if initial_credit_dist_match:
                    parsed_data['initial_credit_distribution'] = eval(initial_credit_dist_match.group(1))
                if final_credit_dist_match:
                    parsed_data['final_credit_distributions'] = eval(final_credit_dist_match.group(1))
                if recall_match:
                    parsed_data['recall'] = eval(recall_match.group(1))
                if precision_match:
                    parsed_data['precision'] = eval(precision_match.group(1))
                if profit_rate_match:
                    parsed_data['profit_rate'] = eval(profit_rate_match.group(1))
                if accuracy_match:
                    parsed_data['accuracy'] = eval(accuracy_match.group(1))
                if recall_features_match:
                    parsed_data['recall_features'] = eval(recall_features_match.group(1))
                if precision_features_match:
                    parsed_data['precision_features'] = eval(precision_features_match.group(1))
                # if cumulative_loans_match:
                #     parsed_data['cumulative_loans'] = eval(cumulative_loans_match.group(1))
                # if cumulative_recall_match:
                #     parsed_data['cumulative_recall'] = eval(cumulative_recall_match.group(1))
                
                result[path] = parsed_data
    
    return result


values = parse_text_files(final_files)

#Function that finds the subset of dictionaries in the original dictionary 
# that contain all strings in a given list and not contains any of the strings in another list
def find_subset(dictionary, contains, not_contains):
    result = {}
    
    for key, value in dictionary.items():
        if all(string in key for string in contains) and not any(string in key for string in not_contains):
            result[key] = value
    
    return result


#Plot the profit_rate for each model in a bar plot with the name of the model as x-axis
def plot_profit_rate(dictionary, title, x_label, y_label):
    #Check if path exists, if not create it
    if not os.path.exists("evaluation_plots"):
        os.makedirs("evaluation_plots")

    profit_rates = []
    model_names = []
    
    for key, value in dictionary.items():
        profit_rates.append(value['profit_rate'][0])
        if 'Curiosity'==x_label:
            #Keep only key between first and second underscore
            key = key.split("_", 2)[1]
            model_names.append(key)
        elif 'Model name'==x_label:
            #Change key to part after "final" and before "boosted"
            key = key.split("final", 1)[1]
            key = key.split("boosted", 1)[0]
            model_names.append(key)

    #Sort model_names and keys on value of key, keep the same order for both lists
    model_names, profit_rates = zip(*sorted(zip(model_names, profit_rates)))
    plt.figure(figsize=(10,10))
    plt.title(title)
    plt.bar(model_names, profit_rates)
    plt.xticks(rotation=90)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.savefig(f"evaluation_plots/{title}.png")
    plt.show()

#Define subsets of dictionaries
final_300_UCB = find_subset(values, ['final', '400000', 'UCB'], ['boosted'])
final_300_visit = find_subset(values, ['final', '400000', 'visit_count'], ['boosted'])
final_300_scalar = find_subset(values, ['final', '400000', 'scalar'], ['boosted'])

final_400_models_15 = find_subset(values, ['final', '400000', '_1.5_'], ['boosted'])
final_400_models_05 = find_subset(values, ['final', '400000', '_0.5_'], ['boosted'])
final_400_models_3 = find_subset(values, ['final', '100000', '_3_'], ['boosted'])

# final_300_models_1 = find_subset(values, ['final', '300000', '_1_'], ['boosted'])
# final_200_models_1 = find_subset(values, ['final', '200000', '_1_'], ['boosted'])
# final_100_models_1 = find_subset(values, ['final', '100000', '_1_'], ['boosted'])

# final_300_models_01 = find_subset(values, ['final', '300000', '_0.1_'], ['boosted'])
# final_200_models_01 = find_subset(values, ['final', '200000', '_0.1_'], ['boosted'])
# final_100_models_01 = find_subset(values, ['final', '100000', '_0.1_'], ['boosted'])


#Plot profit rate comparisons for 300000 steps

plot_profit_rate(final_300_UCB, 'Profit Rate for final UCB models with 400000 steps',
                  'Curiosity', 'Profit Rate')
plot_profit_rate(final_300_visit, 'Profit Rate for final visit count models with 400000 steps',
                  'Curiosity', 'Profit Rate')
plot_profit_rate(final_300_scalar, 'Profit Rate for final scalar models with 400000 steps',
                    'Curiosity', 'Profit Rate') 
plot_profit_rate(final_400_models_05, 'Profit Rate for final models with 300000 steps',
                  'Model name', 'Profit Rate')
plot_profit_rate(final_400_models_3, 'Profit Rate for final models with 300000 steps',
                  'Model name', 'Profit Rate')
# plot_profit_rate(final_200_models, 'Profit Rate for final models with 200000 steps',
#                     'Model name', 'Profit Rate')
# plot_profit_rate(final_100_models, 'Profit Rate for final models with 100000 steps',
#                     'Model name', 'Profit Rate')



def plot_final_credit_dist(dictionary, title, x_label, y_label):
    #Check if path exists, if not create it
    if not os.path.exists("evaluation_plots"):
        os.makedirs("evaluation_plots")

    fig, axs = plt.subplots(1, len(dictionary), figsize=(10,5), sharey=True)
    fig.suptitle(title)
    fig.tight_layout(pad=3.0)

    #if x_label is curisotiy sort dictionary on value between underscores
    if x_label == 'curiosity':
        dictionary = dict(sorted(dictionary.items(), key=lambda item: item[0].split("_", 2)[1]))

    for i, (key, value) in enumerate(dictionary.items()):
        axs[i].plot(value['final_credit_distributions']['0'], label='Group 0')
        axs[i].plot(value['final_credit_distributions']['1'], label='Group 1')

        if x_label == 'model':
            key = key.split("final", 1)[1]
            key = key.split("boosted", 1)[0]
            axs[i].set_title(f"Model: {key}")
        elif x_label == 'curiosity':
            key = key.split("_", 2)[1]
            axs[i].set_title(f"Curiosity factor: {key}")
        elif x_label == 'boosted':
            if 'boosted' in key:
                axs[i].set_title(f"Boosted")
            else:
                axs[i].set_title(f"Standard")
        axs[i].set_ylabel(y_label)
        axs[i].set_xlabel("Population group")
        axs[i].legend()
    #Grid
    for ax in axs.flat:
        ax.grid(True)
    plt.savefig(f"evaluation_plots/{title}.png")
    plt.show()

#Plot final credit distributions

plot_final_credit_dist(final_300_UCB, 'Final credit distribution for final UCB models with 400000 steps', 
                       x_label='curiosity', y_label='Credit distribution'),
plot_final_credit_dist(final_300_visit, 'Final credit distribution for final visit count models with 400000 steps',
                          x_label='curiosity', y_label='Credit distribution')
plot_final_credit_dist(final_300_scalar, 'Final credit distribution for final scalar models with 400000 steps',
                            x_label='curiosity', y_label='Credit distribution')


# plot_final_credit_dist(final_300_models_2, 'Final credit distribution for final models with 300000 steps and curiosity 2',
#                           x_label='model', y_label='Credit distribution')
# plot_final_credit_dist(final_200_models_2, 'Final credit distribution for final models with 200000 steps and curiosity 2',
#                             x_label='model', y_label='Credit distribution')
# plot_final_credit_dist(final_100_models_2, 'Final credit distribution for final models with 100000 steps and curiosity 2',
#                             x_label='model', y_label='Credit distribution')

# plot_final_credit_dist(final_300_models_1, 'Final credit distribution for final models with 300000 steps and curiosity 1',
#                             x_label='model', y_label='Credit distribution')
# plot_final_credit_dist(final_200_models_1, 'Final credit distribution for final models with 200000 steps and curiosity 1',
#                             x_label='model', y_label='Credit distribution')
# plot_final_credit_dist(final_100_models_1, 'Final credit distribution for final models with 100000 steps and curiosity 1',
#                             x_label='model', y_label='Credit distribution')

# plot_final_credit_dist(final_300_models_01, 'Final credit distribution for final models with 300000 steps and curiosity 0.1',
#                             x_label='model', y_label='Credit distribution')
# plot_final_credit_dist(final_200_models_01, 'Final credit distribution for final models with 200000 steps and curiosity 0.1', 
#                             x_label='model', y_label='Credit distribution')
# plot_final_credit_dist(final_100_models_01, 'Final credit distribution for final models with 100000 steps and curiosity 0.1',
#                             x_label='model', y_label='Credit distribution')


boosted_UCB_300_2 = find_subset(values, ['UCB', '400000', '_3_', 'final'], [])
boosted_visit_300_2 = find_subset(values, ['visit', '400000', '_3_', 'final'], [])
boosted_scalar_300_2 = find_subset(values, ['scalar', '400000', '_3_', 'final'], [])

#Sort above dicts on key reversed alphabetically
boosted_UCB_300_2 = dict(sorted(boosted_UCB_300_2.items(), key=lambda item: item[0]))
boosted_visit_300_2 = dict(sorted(boosted_visit_300_2.items(), key=lambda item: item[0]))
boosted_scalar_300_2 = dict(sorted(boosted_scalar_300_2.items(), key=lambda item: item[0]))

plot_final_credit_dist(boosted_UCB_300_2, 'Final boosted credit distribution for UCB models with 400000 steps and curiosity 3',
                            x_label='boosted', y_label='Credit distribution')
plot_final_credit_dist(boosted_visit_300_2, 'Final boosted credit distribution for visit count models with 400000 steps and curiosity 3',
                            x_label='boosted', y_label='Credit distribution')
plot_final_credit_dist(boosted_scalar_300_2, 'Final boosted credit distribution for scalar models with 400000 steps and curiosity 3',
                            x_label='boosted', y_label='Credit distribution')