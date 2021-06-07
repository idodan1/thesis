from class_NN import *
from RandomForest import *
from LinearRegression import *
from class_CONV import *
from class_conv_img import *
from RandomModel import *


def add_random_model_results():
    cols_res_df = ['target', "model", 'features', 'validation R^2', 'validation RMSE']
    random_model_results_master = {'clay_m': 3.3, 'silt_m': 3.06, 'sand_m': 3.55, 'sum rmse': 9.91}
    random_model_results_hydro = {'clay_h': 4.77, 'silt_h': 4.12, 'sand_h': 4.51, 'sum rmse': 13.69}
    for texture_name in ['master sizer', 'hydro meter']:
        curr_dict = random_model_results_master if texture_name == 'master sizer' else random_model_results_hydro
        results_df_name = 'results_all_models/results all models-{0}.xlsx'
        results_df = pd.read_excel(results_df_name.format(texture_name))
        with open('texture cols {0}'.format(texture_name), 'rb') as f:
            texture_cols = pickle.load(f)
        texture_cols.append('sum rmse')
        for col in texture_cols:
            res_dict = {'target': [col], 'texture type': [texture_name], "model": ['Random model'],
                        'features': "", 'validation R^2':
                            [""], 'validation RMSE': [curr_dict[col]]}
            row = pd.DataFrame.from_dict(res_dict)
            results_df = pd.concat([results_df, row], sort=True)
            results_df = results_df.sort_values(by=['target', 'model'])
            temp = pd.DataFrame()
            for col in cols_res_df:
                temp[col] = results_df[col]
            results_df = temp
            results_df_name = 'results_all_models/results all models-{0} with random.xlsx'
            results_df.to_excel(results_df_name.format(texture_name), index=False)


def main():
    cols_res_df = ['target', "model", 'features', 'validation R^2', 'validation RMSE']
    results_dir = 'results_all_models/'
    results_df_name = 'results_all_models/results all models-{0}.xlsx'
    models_path = 'results_all_models/models/{0}-{1}'

    models_reg = []
    # models_reg = [RandomForest, LinearReg]
    models_features = [Conv, Conv_img, NN]
    models = models_reg + models_features
    sum_rmse_index = -1

    # for texture_name in ['master sizer', 'hydro meter']:
    for texture_name in ['hydro meter']:
        compare_results = False
        results_df, file_exist = get_results_df(results_df_name.format(texture_name), cols_res_df)
        """
        the validation set is coming from the test set which is not the right thing to do, we need
        to change this in all the files so when the code is presented to someone he won't notice that
        """
        train_df, test_df, texture_cols, cols_for_model = get_train_test_cols(texture_name)
        train_x, val_x, test_x = train_df[cols_for_model], test_df[cols_for_model], test_df[cols_for_model]
        train_y, val_y, test_y = train_df[texture_cols], test_df[texture_cols], test_df[texture_cols]
        test_y_lst = [list(rows.values) for index, rows in test_y.iterrows()]
        train_y_lst = [list(rows.values) for index, rows in train_y.iterrows()]

        for k in range(len(models)):
            last_sum_rmse = np.inf
            model_type = models[k]
            model_name = model_type.model_name

            if model_type in models_features:
                features = get_best(model_name, texture_name)
                model = model_type(train_x, val_x, test_x, train_y, val_y, test_y, features)
            else:
                model = model_type(train_x, test_x, train_y, test_y)

            model.create_model()
            model.train()
            predictions = model.predict_test()
            predictions_train = model.predict_train()
            all_rmse = calc_rmses(predictions, test_df, texture_cols)

            if file_exist:
                last_sum_rmse = results_df.loc[results_df['target'] == "sum rmse"]
                last_sum_rmse = last_sum_rmse.loc[last_sum_rmse['model'] == model_name]
                last_sum_rmse = last_sum_rmse['validation RMSE'].values[0]
                compare_results = True

            if last_sum_rmse > all_rmse[sum_rmse_index]:
                model.save(models_path.format(model_name, texture_name))
                if compare_results:
                    indexes = results_df.loc[results_df['model'] == model_name].index
                    results_df = results_df.drop(index=indexes)
                fig, axes = plt.subplots(3, 1)
                for i in range(3):
                    curr_y = [test_y_lst[j][i] for j in range(len(test_y_lst))]
                    curr_p = [predictions[j][i] for j in range(len(predictions))]
                    curr_y_train = [train_y_lst[j][i] for j in range(len(train_y_lst))]
                    curr_p_train = [predictions_train[j][i] for j in range(len(predictions_train))]
                    min_val_x, max_val_x, min_val_y, max_val_y = min(curr_p+curr_p_train), max(curr_p+curr_p_train), \
                                                                 min(curr_y+curr_p), max(curr_y+curr_p)
                    axes[i].plot([min_val_x, max_val_x], [min_val_y, max_val_y])
                    axes[i].scatter(curr_p, curr_y, color='g')
                    axes[i].scatter(curr_p_train, curr_y_train, color='b', marker='+')

                    r_squared = calculate_r_square(curr_p, curr_y, (max_val_y+min_val_y)/2)

                    rmse_col = all_rmse[i]
                    rmse_col = int(rmse_col*100)/100
                    res_str = "r squared = {0}\n RMSE = {1}".format(r_squared, rmse_col)

                    axes[i].plot([], [], ' ',  label=res_str)
                    axes[i].set_xlabel(texture_cols[i] + " predicted")
                    axes[i].set_ylabel(texture_cols[i] + ' real value')
                    axes[i].legend()

                    res_dict = {'target': [texture_cols[i]], "model": [model_name],
                                'features': [str(cols_for_model)],  'validation R^2':
                                    [r_squared], 'validation RMSE': [rmse_col]}
                    row = pd.DataFrame.from_dict(res_dict)
                    results_df = pd.concat([results_df, row], sort=True)

                sum_rmse = all_rmse[sum_rmse_index]
                sum_rmse = int(sum_rmse * 100) / 100
                res_dict = {'target': ['sum rmse'], 'texture type': [texture_name], "model": [model_name],
                            'features': [str(cols_for_model)], 'validation R^2':
                                [""], 'validation RMSE': [sum_rmse]}
                row = pd.DataFrame.from_dict(res_dict)
                results_df = pd.concat([results_df, row], sort=True)
                plt.suptitle(model_name)
                plt.savefig(results_dir + '/images/{0} {1}'.format(model_name, texture_name))
                # plt.show()


        temp = pd.DataFrame()
        for col in cols_res_df:
            temp[col] = results_df[col]
        results_df = temp
        results_df = results_df.sort_values(by=['target', 'model'])
        results_df.to_excel(results_df_name.format(texture_name), index=False)


# main()
add_random_model_results()


