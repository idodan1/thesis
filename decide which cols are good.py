import glob
import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np


def find_losses(col, current_cols, features_n_lost):
    feature = 0
    loss = 1
    loss_with = 0
    loss_without = 0
    for item in features_n_lost:
        if set(item[feature]) == set(current_cols):
            loss_without = item[loss]
        if set(item[feature]) == set(current_cols+[col]):
            loss_with = item[loss]
    return loss_with, loss_without


def main():
    cols_for_model = ['ECaV_2019', 'ECaV_man', 'ECaV_2020', 'ECaV_2018', 'DTM', 'mean_slope', 'NDVI_2_2019', 'NDVI_12_2018']
    result_dir_name = 'results_all_models/'
    models_names = ['linear reg/', 'random_forest/']
    res_dir = 'images_choosing_cols/'
    dir_names = [result_dir_name+name for name in models_names]
    with_f = 0
    without_f = 1
    counter_with = 2
    counter_without = 3
    won = 0
    loser = 1
    font_size = 6
    for texture_type in ['master sizer', 'hydro meter']:
        performance_dict = {col: [0, 0, 0, 0] for col in cols_for_model}
        binary_dict = {col: [0, 0] for col in cols_for_model}
        for d in dir_names:
            curr_files = glob.glob(d + "*")
            curr_files = [f_name for f_name in curr_files if texture_type in f_name and 'division: random' in f_name]
            for file in curr_files:
                df = pd.read_excel(file)[['features', 'total loss']]
                for index, row in df.iterrows():
                    features_list = list(ast.literal_eval(row['features']))
                    features_not_in_list = list(set(cols_for_model) - set(features_list))
                    for f in features_list:
                        performance_dict[f][with_f] += row['total loss']
                        performance_dict[f][counter_with] += 1
                    for f in features_not_in_list:
                        performance_dict[f][without_f] += row['total loss']
                        performance_dict[f][counter_without] += 1


        results_with = [performance_dict[feat][with_f]/performance_dict[feat][counter_with] for feat in performance_dict]
        results_without = [performance_dict[feat][without_f]/performance_dict[feat][counter_without]
                           for feat in performance_dict]

        N = len(cols_for_model)
        ind = np.arange(N)  # the x locations for the groups
        width = 0.27       # the width of the bars

        fig = plt.figure()
        ax = fig.add_subplot(111)

        rects1 = ax.bar(ind, results_with, width, color='r')
        rects2 = ax.bar(ind+width, results_without, width, color='g')

        ax.set_ylabel('total score')
        ax.set_xticks(ind+width)
        ax.set_xticklabels(cols_for_model, fontsize=font_size)
        ax.legend((rects1[0], rects2[0]), ('with', 'without'))
        ax.set_title('feature investigation - compering subsets with against subsets without\n lower bar means small loss means'
                     ' better results - ' + texture_type)
        plt.savefig(res_dir + 'mean losses with or without ' + texture_type)
        # plt.show()
        plt.clf()

        for d in dir_names:
            curr_files = glob.glob(d + "*")
            curr_files = [f_name for f_name in curr_files if texture_type in f_name and 'division: random' in f_name]
            for file in curr_files:
                features_n_lost = []
                df = pd.read_excel(file)[['features', 'total loss']]
                for index, row in df.iterrows():
                    features = list(ast.literal_eval(row['features']))
                    lost = row['total loss']
                    features_n_lost.append([features, lost])
                for col in cols_for_model:
                    other_cols = list(set(cols_for_model) - set([col]))
                    for i in range(1, 2 ** len(other_cols)):
                        num_binary = "{0:b}".format(i)
                        current_permutation = list(num_binary)
                        current_permutation = ['0'] * (len(other_cols) - len(current_permutation)) + current_permutation
                        current_cols = [other_cols[i] for i in range(len(other_cols)) if current_permutation[i] == "1"]
                        loss_with, loss_without = find_losses(col, current_cols, features_n_lost)
                        if loss_with > loss_without:  # if the lost without him is lower it means he lost
                            binary_dict[col][loser] += 1
                        else:
                            binary_dict[col][won] += 1

        results_won = [binary_dict[feat][won] for feat in cols_for_model]
        results_lost = [binary_dict[feat][loser] for feat in cols_for_model]

        N = len(cols_for_model)
        ind = np.arange(N)  # the x locations for the groups
        width = 0.27       # the width of the bars

        fig = plt.figure()
        ax = fig.add_subplot(111)

        rects1 = ax.bar(ind, results_won, width, color='r')
        rects2 = ax.bar(ind+width, results_lost, width, color='g')

        ax.set_ylabel('counter')
        ax.set_xticks(ind+width)
        ax.set_xticklabels(cols_for_model, fontsize=font_size)
        ax.legend((rects1[0], rects2[0]), ('won', 'lost'))
        ax.set_title('for every feature we checked if a subset was \n better with him or without him - ' + texture_type)

        plt.savefig(res_dir + 'won or lost comparison ' + texture_type)
        # plt.show()


main()




























