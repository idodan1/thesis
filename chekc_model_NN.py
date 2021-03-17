from class_NN import *
import ast


def get_top_permutations(num_of_permutations=3):
    dir_name = 'results_all_models/'
    models = ['linear reg', "random_forest"]
    all_features = []
    for model in models:
        model_dir = dir_name + model + "/"
        master = model_dir + "res df_master sizer.xlsx"
        hydro = model_dir + "res df_hydro meter.xlsx"
        master_df = pd.read_excel(master, index=False)[:num_of_permutations]
        hydro_df = pd.read_excel(hydro, index=False)[:num_of_permutations]
        master_features = set([col for col in list(master_df['features'].values)])
        hydro_features = set([col for col in list(hydro_df['features'].values)])
        both = list(master_features.union(hydro_features))
        all_features += both
    all_features = list(set(all_features))
    all_features = [ast.literal_eval(f) for f in all_features]
    return all_features


results_dir = 'results_all_models/'
cols_model_dict = {'MSaV_2020': 0, 'ECaV_2019': 0, 'TIR_3_2020': 1, 'MSaV_man': 0,
                   'water_content_g': 0, 'max_slope': 1, 'local_elevation': 1,
                   'MSaH_man': 0, 'mzs_eca19': 0, 'water_content_f': 0, 'ECaH_2019': 1,
                   'MSaV_2019': 0, 'ECaH_2018': 1, 'DTM': 1, 'MSaH_2019': 0, 'ECaH_man': 1,
                   'mzs_diff1820': 0, 'ECaH_2020': 1, 'ECaV_man': 0, 'ECaV_2020': 0, 'MSaH_2020': 0,
                   'NDVI_2_2019': 1, 'MSaH_2018': 0, 'flow_distance': 1, 'mean_slope': 1,
                   'flow_accumulation': 1, 'MSaV_2018': 0, 'mzs_eca20': 0, 'ECaV_2018': 0,
                   'NDVI_12_2018': 1, 'aspect': 1, 'TIR_2_2020': 1}

all_data_df, train_nums, val_nums, test_nums, cols_for_model, texture_master_cols,\
texture_hydro_cols, activation_funcs = create_df_n_lists_for_model()

# train_nums = list(set(train_nums) - set(val_nums))

num_of_neurons = [5, 5, 5]
activation_nums = [1, 1, 1]
permutation_to_check = get_top_permutations()
# for texture_name in ['master sizer', 'hydro meter']:
for texture_name in ['hydro meter']:
    texture_cols = texture_master_cols if texture_name == 'master sizer' else texture_hydro_cols
    cols_for_model = ['ECaV_2019', 'ECaV_man', 'ECaV_2020', 'ECaV_2018', 'DTM', 'mean_slope', 'NDVI_2_2019', 'NDVI_12_2018']
    model_name = 'NN'
    cols_for_res_df = ['total loss', 'texture type', "num of features", 'permutation num', 'features']
    res_dir_name = results_dir + model_name
    res_df_name = res_dir_name + "/res df_{0}.xlsx".format(texture_name)
    dir_name = 'results_all_models/' + model_name + "/*"
    res_df = pd.DataFrame(columns=cols_for_res_df)

    for i in range(len(permutation_to_check)):
        if i % 2 == 0:
            print('iter num: {0}'.format(i))
        current_cols = permutation_to_check[i]
        train_df = all_data_df.ix[train_nums]
        val_df = all_data_df.ix[val_nums]
        test_df = all_data_df.ix[test_nums]
        train_x, val_x, test_x = train_df[current_cols], val_df[current_cols], test_df[current_cols]
        train_y, val_y, test_y = train_df[texture_cols], val_df[texture_cols], test_df[texture_cols]

        # model = RandomForest(train_x, test_x, train_y, test_y)
        model = NN(train_x, val_x, test_x, train_y, val_y, test_y)

        model.create_model(num_of_neurons, activation_nums, activation_funcs)
        model.train()
        predictions = model.predict_test()
        loss = model.calc_loss(predictions)

        res_dict = {'total loss': [loss], 'texture type': [texture_name], "num of features": [len(current_cols)],
                    'permutation num': [i], 'features': [str(current_cols)]}
        row = pd.DataFrame.from_dict(res_dict)
        res_df = pd.concat([res_df, row], sort=True)

    res_df = res_df.sort_values(by=['total loss'])
    res_df.to_excel(res_df_name, index=False)


