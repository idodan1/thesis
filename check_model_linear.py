from LinearRegression import *
import ast
import glob


def get_features():
    df = pd.read_excel('results_all_models/random_forest/res df_hydro meter.xlsx')
    cols = list(df['features'].values)
    cols = [ast.literal_eval(col) for col in cols]
    return cols


def create_res_df(df_name, model_name, cols):
    dir_name = 'results_all_models/' + model_name + "/*"
    files = glob.glob(dir_name)
    file_name = df_name
    if file_name not in files:
        df = pd.DataFrame(columns=cols)
        df.to_excel(file_name)


def main():
    features_list = get_features()
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
    texture_name = 'hydro meter'
    texture_cols = texture_master_cols if texture_name == 'master sizer' else texture_hydro_cols

    for cols_for_model in features_list:
        train_df = all_data_df.ix[train_nums]
        test_df = all_data_df.ix[test_nums]
        # cols_for_model = [col for col in cols_model_dict if cols_model_dict[col] == 1]
        train_x, test_x = train_df[cols_for_model], test_df[cols_for_model]
        train_y, test_y = train_df[texture_cols], test_df[texture_cols]

        model_name = 'linear reg'
        cols_for_res_df = ['total loss', 'texture type', 'features']
        res_dir_name = results_dir + model_name
        res_df_name = res_dir_name + "/res df_{0}.xlsx".format(texture_name)
        create_res_df(res_df_name, model_name, cols_for_res_df)
        res_df = pd.read_excel(res_df_name)

        model = LinearReg(train_x, test_x, train_y, test_y)
        model.create_model()
        model.train()
        predictions = model.predict_test()
        loss = model.calc_loss(predictions)

        if len(res_df) < 20:
            res_dict = {'total loss': [loss], 'texture type': [texture_name], 'features': [str(cols_for_model)]}
            row = pd.DataFrame.from_dict(res_dict)
            res_df = pd.concat([res_df, row], sort=True)
            res_df = res_df.sort_values(by=['total loss'])
        else:
            max_val = max(list(res_df['total loss'].values))
            if max_val > loss:
                index = list(res_df.loc[res_df['total loss'] == max_val].index)[0]
                res_df.loc[index, 'total loss'] = loss
                res_df.loc[index, 'features'] = str(cols_for_model)
                res_df = res_df.sort_values(by=['total loss'])

        res_df.to_excel(res_df_name, index=False)

main()

