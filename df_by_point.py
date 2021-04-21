from class_NN import *
from RandomForest import *
from LinearRegression import *
from class_CONV import *
from class_conv_img import *


def label_sample(row, texture_cols):
    classes = ['sand', 'silt', 'clay']
    res = dict()
    for col in texture_cols:
        for c in classes:
            if c in col:
                res[c] = row[col]
    return to_soil_class(res)


def main():
    cols_res_df = ['target', "model", 'features', 'validation R^2', 'validation RMSE']
    results_dir = 'results_all_models/'
    results_df_name = 'results_all_models/results all models-{0}.xlsx'
    models_path = 'results_all_models/models/{0}-{1}'
    texture_names = ['master sizer', 'hydro meter']
    models_reg = [RandomForest, LinearReg]
    # models_features = [Conv, Conv_img, NN]
    models_features = []
    models = models_reg + models_features

    for texture_name in texture_names:
        results_df, file_exist = get_results_df(results_df_name.format(texture_name), cols_res_df)

        train_df, test_df, texture_cols, cols_for_model = get_train_test_cols(texture_name)
        train_x, val_x, test_x = train_df[cols_for_model], test_df[cols_for_model], test_df[cols_for_model]
        train_y, val_y, test_y = train_df[texture_cols], test_df[texture_cols], test_df[texture_cols]
        test_y_lst = [list(rows.values) for index, rows in test_y.iterrows()]
        train_y_lst = [list(rows.values) for index, rows in train_y.iterrows()]
        # add class to df
        soil_class_name = 'soil_class_{0}'.format(texture_name)
        train_y[soil_class_name] = train_y.apply(lambda row: label_sample(row, texture_cols), axis=1)
        test_y[soil_class_name] = test_y.apply(lambda row: label_sample(row, texture_cols), axis=1)
        # all_y = pd.concat([train_y, test_y])
        all_y = test_y
        # all_x = pd.concat(([train_x, test_x]))
        all_x = test_x

        for model_type in models:
            model_name = model_type.model_name

            if model_type in models_features:
                features = get_best(model_name, texture_name)
                model = model_type(train_x, val_x, test_x, train_y, val_y, test_y, features)
            else:
                model = model_type(train_x, test_x, train_y, test_y)
            model.load(models_path.format(model_name, texture_name))

            all_predictions = []
            for index, row in all_y.iterrows():
                values = all_x.loc[index].values
                prediction = model.predict(values)
                all_predictions.append(prediction)
            prediction_texture_col_name = '{0}-{1}'
            rmse_texture_col_name = 'rmse-{0}-{1}'
            letter_to_add = texture_name[0]
            for i in range(len(texture_cols)):
                all_y[prediction_texture_col_name.format(texture_cols[i], model_name)] = [p[i] for p in all_predictions]
                all_y[rmse_texture_col_name.format(texture_cols[i], model_name)] = abs(all_y[prediction_texture_col_name.
                    format(texture_cols[i], model_name)] - all_y[texture_cols[i]])
            all_y[rmse_texture_col_name.format('sum_' + letter_to_add, model_name)] = 0
            for t in texture_cols:
                all_y[rmse_texture_col_name.format('sum_'+letter_to_add, model_name)] += all_y[rmse_texture_col_name.format(t, model_name)]
        models_names = [m.model_name for m in models]
        plot_point_error(all_y, models_names, texture_name, texture_cols)


def plot_point_error(all_y, models_names, texture_name, texture_cols):
    # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
    # shape = ['x', 'o']
    # lines = []
    # rmse_texture_col_name = 'rmse-{0}-{1}'
    # x_values = list(all_y.index)
    # letter_to_add = texture_name[0]
    # colors = ['r', 'g', 'b']
    # for i in range(len(models_names)):
    #     model_name = models_names[i]
    #     y_values = all_y[rmse_texture_col_name.format('sum_' + letter_to_add, model_name)].values
    #     line = axes.scatter(x_values, y_values, c=colors[i], label=model_name)
    #     lines.append(line)
    # axes.set_xlabel('point index')
    # axes.set_ylabel('sum rmse')
    # axes.legend(handles=lines, loc='upper right', bbox_to_anchor=(0.95, 0.8))
    # axes.set_title(texture_name + ' sum rmse per point')
    # plt.show()
    #
    rmse_texture_col_name = 'rmse-{0}-{1}'
    models_name = models_names[0]
    plot_df = all_y[[rmse_texture_col_name.format(c, models_name) for c in texture_cols]]
    plot_df.plot(kind='bar', stacked=True)
    plt.title(texture_name + ' rmse per point')
    plt.xlabel('point number')
    plt.ylabel('rmse')
    plt.show()


main()
# https://stackoverflow.com/questions/39013425/multiple-stacked-bar-plot-with-pandas


"""
we want to create a df of the texture data by point that has the real value and the prediction value of every model and
maybe the class. we will use this results to do a bar graph that shows the rmse of every test point (and maybe train).


"""
