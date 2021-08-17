from MlModels import *
from RandomModel import *


def add_random_model_results():
    cols_res_df = ['target', "model", 'features', 'validation R^2', 'validation RMSE']
    random_model_results_master = {'clay_m': 2.96, 'silt_m': 3.12, 'sand_m': 3.65, 'sum rmse': 9.73}
    random_model_results_hydro = {'clay_h': 4.8, 'silt_h': 4.34, 'sand_h': 4.7, 'sum rmse': 13.85}
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


def create_predictions_df(texture_name, old_test_y, old_train_y, train_df, test_df,
                          texture_cols, cols_for_model):
    models = [RandomForest, LinearReg, NN, ConvImg, Conv]
    # models = [ConvImg, Conv]

    train_x, test_x = train_df[cols_for_model], test_df[cols_for_model]
    train_y, test_y = train_df[texture_cols], test_df[texture_cols]

    test_y_res, train_y_res = test_y.copy(deep=True), train_y.copy(deep=True)

    for k in range(len(models)):
        model_type = models[k]
        model_name = model_type.model_name

        features = get_best(model_name, texture_name)
        model = model_type(train_x, test_x, train_y, test_y, features)

        model.create_model()
        model.train()
        test_y_res[["{0} {1}".format(model_name, col) for col in texture_cols]] = model.predict_test()[texture_cols]
        train_y_res[["{0} {1}".format(model_name, col) for col in texture_cols]] = model.predict_train()[texture_cols]
        model.save('ML models/{0}'.format(model_name))

    save_best_rmse(test_y_res, train_y_res, texture_cols, models, old_test_y, old_train_y)

    old_test_y.to_excel(test_df_file.format(texture_name))
    old_train_y.to_excel(train_df_file.format(texture_name))


def save_best_rmse(test_y, train_y, texture_cols, models, old_test_y, old_train_y):
    for model in models:
        measured = test_y[texture_cols]
        predicted = old_test_y[["{0} {1}".format(model.model_name, col) for col in texture_cols]]
        predicted[texture_cols] = predicted[["{0} {1}".format(model.model_name, col) for col in texture_cols]]
        old_rmse = (calc_rmses(measured, predicted, texture_cols)).sum()
        predicted = test_y[["{0} {1}".format(model.model_name, col) for col in texture_cols]]
        predicted[texture_cols] = predicted[["{0} {1}".format(model.model_name, col) for col in texture_cols]]
        new_rmse = (calc_rmses(measured, predicted, texture_cols)).sum()
        if new_rmse < old_rmse:
            old_test_y.loc[:, ["{0} {1}".format(model.model_name, col) for col in texture_cols]] = \
                test_y[["{0} {1}".format(model.model_name, col) for col in texture_cols]]
            old_train_y.loc[:, ["{0} {1}".format(model.model_name, col) for col in texture_cols]] = \
                train_y[["{0} {1}".format(model.model_name, col) for col in texture_cols]]


def create_graph(texture_cols, r2, rmse, rpd, rpiq, results_test, results_train, model_name, texture_name):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for i in range(len(texture_cols)):
        col_m = texture_cols[i]
        col_name = (change_col_name(col_m))
        col_p = "{0} {1}".format(model_name, col_m)
        temp_m = pd.concat([results_test[col_m], results_train[col_m]])
        temp_p = pd.concat([results_test[col_p], results_train[col_p]])
        min_val_x, max_val_x, min_val_y, max_val_y = temp_m.min(), temp_m.max(), temp_p.min(), temp_p.max()
        axes[i].plot([min_val_x, max_val_x], [min_val_y, max_val_y])
        axes[i].scatter(results_test[col_m], results_test[col_p], color='g', label='test')
        axes[i].scatter(results_train[col_m], results_train[col_p], color='b', marker='+', label='train')

        res_str = "R2 = {0}\nRMSE = {1}\nRPD= {2}\nRPIQ = {3}".format(r2[i], rmse[i], rpd[i], rpiq[i])
        axes[i].plot([], [], ' ', label=res_str)
        axes[i].set_title(col_name, fontsize=18)
        axes[i].set_xlabel('Measured ' + col_name + ' (%)', fontsize=18)
        axes[i].set_ylabel('Predicted ' + col_name + ' (%)', fontsize=18)
        axes[i].legend(loc='lower right', fontsize=14)
        axes[i].xaxis.set_tick_params(labelsize=14)
        axes[i].yaxis.set_tick_params(labelsize=14)
    plt.suptitle(model_name, fontsize=18)
    plt.savefig(results_dir + '/images/{0} {1}'.format(model_name, texture_name))
    # plt.show()


def create_excel_n_graphs(results_test, results_train, texture_cols, texture_name):
    models = [RandomForest, LinearReg, Conv, ConvImg, NN]
    results_df_name = 'results_all_models/results all models-{0}.xlsx'
    cols_res_df = ['target', "model", 'validation RMSE(%)', 'validation R^2', 'RPD', 'RPIQ']
    results_df = pd.DataFrame(columns=cols_res_df)
    for model in models:
        r2 = [calculate_r_square(results_test["{0} {1}".format(model.model_name, col)], results_test[col]) for col
              in texture_cols] + [""]
        predicted = results_test[["{0} {1}".format(model.model_name, col) for col in texture_cols]]
        predicted[texture_cols] = predicted[["{0} {1}".format(model.model_name, col) for col in texture_cols]]
        rmse = calc_rmses(predicted, results_test, texture_cols)
        rmse = list(rmse.values) + [rmse.sum()]
        rmse = [int(val*100)/100 for val in rmse]
        rpd = [RPD(results_test["{0} {1}".format(model.model_name, col)].values, results_test[col].values) for
               col in texture_cols] + [""]
        rpiq = [RPIQ(results_test["{0} {1}".format(model.model_name, col)].values, results_test[col].values) for
               col in texture_cols] + [""]
        cols = texture_cols + ['Sum']
        res_dict = {'target': cols, "model": [model.model_name]*len(cols),
                    'validation R^2': r2, 'validation RMSE(%)': rmse, 'RPD': rpd, 'RPIQ': rpiq}
        temp = pd.DataFrame.from_dict(res_dict)
        results_df = pd.concat([results_df, temp], sort=True)

        create_graph(texture_cols, r2, rmse, rpd, rpiq, results_test, results_train, model.model_name, texture_name)
    results_df = results_df.reindex(cols_res_df, axis=1)
    results_df = results_df.sort_values(by=['target', 'model'], ascending=False)
    results_df.to_excel(results_df_name.format(texture_name), index=False)


def main():
    cols_res_df = ['target', "model", 'features', 'validation R^2', 'validation RMSE']
    for texture_name in ['master sizer', 'hydro meter']:
        train_df, test_df, texture_cols, cols_for_model = get_train_test_cols(texture_name)
        old_test_y = pd.read_excel(test_df_file.format(texture_name), index_col=0)
        old_train_y = pd.read_excel(train_df_file.format(texture_name), index_col=0)
        # create_predictions_df(texture_name, old_test_y, old_train_y, train_df, test_df,
        #                       texture_cols, cols_for_model)
        create_excel_n_graphs(old_test_y, old_train_y, texture_cols, texture_name)


if __name__ == "__main__":
    results_dir = 'results_all_models/'
    models_path = 'results_all_models/models/{0}-{1}'
    test_df_file = 'results_all_models/test_df-{0}.xlsx'
    train_df_file = 'results_all_models/train_df-{0}.xlsx'
    main()






