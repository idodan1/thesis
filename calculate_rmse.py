from class_NN import *
from RandomForest import *
from LinearRegression import *
from class_CONV import *
from class_conv_img import *
from RandomModel import *


def main():
    cols_res_df = ['target', "model", 'features', 'validation R^2', 'validation RMSE']
    results_dir = 'results_all_models/'
    results_df_name = 'results_all_models/results all models-{0}.xlsx'
    models_path = 'results_all_models/models/{0}-{1}'

    models_reg = [RandomForest, LinearReg]
    models_features = [Conv, Conv_img, NN]
    models = models_reg + models_features
    sum_rmse_index = -1

    for texture_name in ['master sizer', 'hydro meter']:
    # for texture_name in ['master sizer']:
        compare_results = False
        results_df, file_exist = get_results_df(results_df_name.format(texture_name), cols_res_df)

        train_df, test_df, texture_cols, cols_for_model = get_train_test_cols(texture_name)
        train_x, val_x, test_x = train_df[cols_for_model], test_df[cols_for_model], test_df[cols_for_model]
        train_y, val_y, test_y = train_df[texture_cols], test_df[texture_cols], test_df[texture_cols]
        test_y_lst = [list(rows.values) for index, rows in test_y.iterrows()]
        train_y_lst = [list(rows.values) for index, rows in train_y.iterrows()]

        for k in range(len(models)):
            last_sum_rmse = np.inf
            model_type = models[k]
            model_name = model_type.model_name
            iterations = 1 if model_type in models_features else 1

            for h in range(iterations):
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
                    previous_index = last_sum_rmse.index[0]
                    last_sum_rmse = last_sum_rmse['validation RMSE'].values[0]
                    compare_results = True

                if last_sum_rmse > all_rmse[sum_rmse_index]:
                    model.save(models_path.format(model_name, texture_name))
                    if compare_results:
                        results_df = results_df.drop(index=[previous_index])
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
        results_df = results_df.sort_values(by=['model'])
        results_df = results_df.sort_values(by=['target'])
        results_df.to_excel(results_df_name.format(texture_name), index=False)


main()


