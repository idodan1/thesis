from RandomForest import *
from LinearRegression import *

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

for texture_name in ['master sizer', 'hydro meter']:
# texture_name = 'master sizer'
    texture_cols = texture_master_cols if texture_name == 'master sizer' else texture_hydro_cols
    cols_for_model = ['ECaV_2019', 'ECaV_man', 'ECaV_2020', 'ECaV_2018', 'DTM', 'mean_slope', 'NDVI_2_2019', 'NDVI_12_2018']
    # model_name = 'random_forest'
    model_name = 'linear reg'
    cols_for_res_df = ['total loss', 'texture type', "num of features", 'permutation num', 'features']
    res_dir_name = results_dir + model_name
    res_df_name = res_dir_name + "/res df_{0}.xlsx".format(texture_name)
    dir_name = 'results_all_models/' + model_name + "/*"
    res_df = pd.DataFrame(columns=cols_for_res_df)

    for i in range(1, 2**len(cols_for_model)):
        if i % 20 == 0:
            print('iter num: {0}'.format(i))
        num_binary = "{0:b}".format(i)
        current_permutation = list(num_binary)
        current_permutation = ['0']*(len(cols_for_model)-len(current_permutation)) + current_permutation
        current_cols = [cols_for_model[i] for i in range(len(cols_for_model)) if current_permutation[i] == "1"]
        train_df = all_data_df.ix[train_nums]
        test_df = all_data_df.ix[test_nums]
        train_x, test_x = train_df[current_cols], test_df[current_cols]
        train_y, test_y = train_df[texture_cols], test_df[texture_cols]

        # model = RandomForest(train_x, test_x, train_y, test_y)
        model = LinearReg(train_x, test_x, train_y, test_y)
        model.create_model()
        model.train()
        predictions = model.predict_test()
        loss = model.calc_loss(predictions)

        res_dict = {'total loss': [loss], 'texture type': [texture_name], "num of features": [len(current_cols)],
                    'permutation num': [i], 'features': [str(current_cols)]}
        row = pd.DataFrame.from_dict(res_dict)
        res_df = pd.concat([res_df, row], sort=True)

    res_df = res_df.sort_values(by=['total loss'])
    res_df.to_excel(res_df_name, index=False)


