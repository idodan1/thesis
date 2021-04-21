import pandas as pd
import matplotlib.pyplot as plt
import ast


dir_name = 'results_all_models/'
models = ['Linear Regression', "Random Forest"]
measuring_methods = ['master sizer', 'hydro meter']
cols_for_model = ['ECaV_2019', 'ECaV_man', 'ECaV_2020', 'ECaV_2018', 'DTM', 'mean_slope', 'NDVI_2_2019', 'NDVI_12_2018']
for model in models:
    model_dir = dir_name + model + "/"
    master = model_dir + "res df_master sizer.xlsx"
    hydro = model_dir + "res df_hydro meter.xlsx"
    master_df = pd.read_excel(master, index=False)
    hydro_df = pd.read_excel(hydro, index=False)
    to_add = 0 if model == 'NN' else 1
    master_values = [list((master_df.loc[master_df['permutation num'] == i])['total loss'].values)[0]
                     for i in range(1, len(master_df) + to_add)]
    hydro_values = [list((hydro_df.loc[hydro_df['permutation num'] == i])['total loss'].values)[0]
                     for i in range(1, len(hydro_df) + to_add)]
    plt.plot([min(hydro_values + master_values), max(hydro_values + master_values)], [min(hydro_values + master_values), max(hydro_values + master_values)])
    plt.scatter(master_values, hydro_values)
    plt.xlabel('master sizer lost')
    plt.ylabel('hydro meter lost')
    plt.title("comparison between measurements methods for {0}".format(model))
    plt.show()
    
    
for method in measuring_methods:
    master = dir_name + "res df_master sizer.xlsx"
    hydro = dir_name + "res df_hydro meter.xlsx"
    linear = dir_name + 'Linear Regression/' + 'res df_{0}.xlsx'.format(method)
    forest = dir_name + 'Random Forest/' + 'res df_{0}.xlsx'.format(method)
    NN = dir_name + 'NN/' + 'res df_{0}.xlsx'.format(method)
    linear_df = pd.read_excel(linear, index=False)
    forest_df = pd.read_excel(forest, index=False)
    NN_df = pd.read_excel(NN, index=False)
    linear_values = [list((linear_df.loc[linear_df['permutation num'] == i])['total loss'].values)[0]
                     for i in range(1, len(linear_df) + 1)]
    forest_values = [list((forest_df.loc[forest_df['permutation num'] == i])['total loss'].values)[0]
                     for i in range(1, len(forest_df) + 1)]
    NN_values = [list((forest_df.loc[forest_df['permutation num'] == i])['total loss'].values)[0]
                     for i in range(1, len(forest_df) + 1)]
    values_for_plot = [linear_values[i]/forest_values[i] for i in range(len(forest_values))]
    plt.scatter(list(range(1, len(values_for_plot)+1)), values_for_plot)
    plt.plot([0, len(values_for_plot)], [1, 1])
    plt.xlabel('permutation num')
    plt.ylabel('(linear lost)/(forest lost)')
    plt.title("comparison between models for {0} measuring method".format(method))
    plt.show()

for model in models:
    num_of_top_rows = 10
    model_dir = dir_name + model + "/"
    master = model_dir + "res df_master sizer.xlsx"
    hydro = model_dir + "res df_hydro meter.xlsx"
    master_df = pd.read_excel(master, index=False)[:num_of_top_rows]
    hydro_df = pd.read_excel(hydro, index=False)[:num_of_top_rows]
    master_permutations = list(master_df['permutation num'].values)
    master_features = [set(ast.literal_eval(col)) for col in list(master_df['features'].values)]
    master_features_in_all = set.union(*master_features)
    hydro_permutations = list(hydro_df['permutation num'].values)
    hydro_features = [set(ast.literal_eval(col)) for col in list(hydro_df['features'].values)]
    hydro_features_in_all = set.union(*hydro_features)
    print('results for model: {0}'.format(model))
    print('permutations that appear in both methods (top {0}): {1}'.format(num_of_top_rows, set(master_permutations)
                                                                           .intersection(set(hydro_permutations))))
    print('features that appear in both methods (top {0}): {1}'.format(num_of_top_rows, set(master_features_in_all)
                                                                       .intersection(set(hydro_features_in_all))))
    print('num of features in top {0} master sizer (out of {1}): {2}'.format(num_of_top_rows, len(cols_for_model), len(master_features_in_all)))
    print('num of features in top {0} hydro meter (out of {1}): {2}'.format(num_of_top_rows, len(cols_for_model), len(hydro_features_in_all)))
    print('features in top {0} master sizer: {1}'.format(num_of_top_rows, master_features_in_all))
    print('features in top {0} hydro meter: {1}'.format(num_of_top_rows, hydro_features_in_all))
    print('features that are not in top {0} master sizer: {1}'.format(num_of_top_rows, set(cols_for_model)
                                                                      - master_features_in_all))
    print('features that are not in top {0} hydro meter: {1}'.format(num_of_top_rows, set(cols_for_model)
                                                                     - hydro_features_in_all))
    print()

# for method in models:
#     num_of_top_rows = 3
#     master = dir_name + "res df_master sizer.xlsx"
#     hydro = dir_name + "res df_hydro meter.xlsx"
#     linear = dir_name + 'Linear Regression/' + 'res df_{0}.xlsx'.format(method)
#     forest = dir_name + 'Random Forest/' + 'res df_{0}.xlsx'.format(method)
#     linear_df = pd.read_excel(linear, index=False)[:num_of_top_rows]
#     forest_df = pd.read_excel(forest, index=False)[:num_of_top_rows]
#     master_permutations = list(master_df['permutation num'].values)
#     master_features = [set(ast.literal_eval(col)) for col in list(master_df['features'].values)]
#     features_in_all = set.union(*master_features)
#     hydro_permutations = list(hydro_df['permutation num'].values)
#     print('results for model: {0}'.format(model))
#     print('common permutations: {0}'.format(set(master_permutations).intersection(set(hydro_permutations))))
#     print('num of features in top {0}: {1}'.format(num_of_top_rows, len(features_in_all)))
#     print('features in top {0}: {1}'.format(num_of_top_rows, features_in_all))
#     print('features that are not in top {0}: {1}'.format(num_of_top_rows, set(cols_for_model) - features_in_all))
#     print()

