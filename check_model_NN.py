from class_NN import *
import ast
import glob


def get_top_permutations(num_of_permutations=3):
    dir_name = 'results_all_models/'
    models = ['linear reg', "random_forest"]
    all_features = set()
    for model in models:
        model_dir = dir_name + model + "/"
        files = glob.glob(model_dir + "*")
        for file in files:
            res_df = pd.read_excel(file, index=False)[:num_of_permutations]
            features = set([col for col in list(res_df['features'].values)])
            all_features = set.union(all_features, features)
    all_features = [ast.literal_eval(f) for f in all_features]
    return all_features


results_dir = 'results_all_models/'
all_data_file = 'soil_data_2020_all data.xlsx'
all_data_df = pd.read_excel(all_data_file, index_col=0)[2:]
cols_for_model_by_texture = [['ECaV_man', 'ECaV_2020', 'DTM', 'mean_slope', 'NDVI_12_2018'], ['ECaV_man', 'ECaV_2020', 'DTM', 'NDVI_2_2019']]
activation_funcs = [activations.sigmoid, activations.relu, activations.linear]
# for option in [' random', '']:
for option in [' random']:
    texture_num = 0
    for texture_name in ['master sizer', 'hydro meter']:
        cols_for_model = cols_for_model_by_texture[texture_num]
        texture_num += 1
        print("currently {0} {1}".format(option, texture_name))
        with open('train nums{0} {1}'.format(option, texture_name), 'rb') as f:
            train_nums = pickle.load(f)
        with open('test nums{0} {1}'.format(option, texture_name), 'rb') as f:
            test_nums = pickle.load(f)
        with open('texture cols {0}'.format(texture_name), 'rb') as f:
            texture_cols = pickle.load(f)

        permutation_to_check = get_top_permutations(num_of_permutations=2)
        model_name = 'NN'
        cols_for_res_df = ['total loss', 'texture type', "num of features", 'division', 'features',
                           "num of neurons", 'activation nums']
        res_dir_name = results_dir + model_name
        res_df_name = res_dir_name + "/res df {0} division: {1}.xlsx".format(texture_name, option)
        dir_name = 'results_all_models/' + model_name + "/*"
        res_df = pd.DataFrame(columns=cols_for_res_df)
        train_df = all_data_df.ix[train_nums]
        test_df = all_data_df.ix[test_nums]
        train_x, val_x, test_x = train_df[cols_for_model], test_df[cols_for_model], test_df[cols_for_model]
        train_y, val_y, test_y = train_df[texture_cols], test_df[texture_cols], test_df[texture_cols]

        num_of_iter = 7
        for activation_num in range(0, 3):
            for num_neurons in range(1, num_of_iter):
                for num_of_layers in range(1, num_of_iter):
                    print('num_neurons num: {0}\n layer type: {1}'.format(num_neurons, num_of_layers))
                    num_of_neurons = [num_neurons]*num_of_layers
                    activation_nums = [activation_num]*num_of_layers

                    model = NN(train_x, val_x, test_x, train_y, val_y, test_y)

                    model.create_model(num_of_neurons, activation_nums)
                    model.train()
                    predictions = model.predict_test()
                    loss = model.calc_loss(predictions)

                    division_type = 'gap statistic' if option == "" else option
                    res_dict = {'total loss': [loss], 'texture type': [texture_name], "num of features": [len(cols_for_model)],
                                'division': [division_type], 'features': [str(cols_for_model)],
                                "num of neurons": [str(num_of_neurons)], 'activation nums': [str(activation_nums)]}
                    row = pd.DataFrame.from_dict(res_dict)
                    res_df = pd.concat([res_df, row], sort=True)

        res_df = res_df.sort_values(by=['total loss'])
        res_df.to_excel(res_df_name, index=False)

"""
- create similar file for conv and perform grid search for image net architecture using the architecture from here
- create dir for results from rmse file and save images and numeric results
- add random model to rmse so we will have a reference result
- find a way to stop training with conv, save the best model and reload it when training is finished
    do it similar to to early stopping
"""