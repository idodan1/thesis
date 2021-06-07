from class_NN import *
from RandomForest import *
from LinearRegression import *
from class_CONV import *
from class_conv_img import *
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


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
    models_features = [Conv, Conv_img, NN]
    # models_features = []
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
        train_y.loc[:, soil_class_name] = train_y.apply(lambda row: label_sample(row, texture_cols), axis=1)
        test_y.loc[:, soil_class_name] = test_y.apply(lambda row: label_sample(row, texture_cols), axis=1)
        all_y = pd.concat([train_y, test_y])
        # all_y = test_y
        all_x = pd.concat(([train_x, test_x]))
        # all_x = test_x
        # create_df_classes(all_y, soil_class_name)
        # plot_textures(test_y, texture_name)

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
                if 'conv' in model_name.lower():
                    image_sizes = [20, 50, 100]
                    image_size = image_sizes[features['mini_image_size_index']]
                    image_name = get_image_name(index)
                    x, y = create_data(image_name, all_y[texture_cols])
                    x_mini = create_mini_images(x, image_size)
                    if model_name == 'Conv':
                        numeric_data = np.array(names_to_arr_num([image_name], all_x)[0])
                        numeric_data = np.array([numeric_data for _ in range(x_mini.shape[0])])
                        prediction, gap = model.predict([x_mini, numeric_data])
                    else:
                        prediction, gap = model.predict(x_mini)
                else:
                    values = list(all_x.loc[index].values)
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
            class_string = 'class-{0}'
            cols = [prediction_texture_col_name.format(texture_cols[i], model_name) for i in range(3)]
            df_for_class = all_y[cols]
            all_y.loc[:, class_string.format(model_name)] = df_for_class.apply(lambda line: label_sample(line, cols), axis=1)
            all_y['right prediction-{0}'.format(model_name)] = np.where(all_y[class_string.format(model_name)] == soil_class_name, 1, 0)
        all_y.to_excel('res by points all models-{0}.xlsx'.format(texture_name), index=False)
        models_names = [m.model_name for m in models]
        # all_y = pd.read_excel('df for bar plot.xlsx')
        # plot_point_error(all_y, models_names, texture_name, texture_cols)


def plot_point_error(all_y, models_names, texture_name, texture_cols):
    rmse_texture_col_name = 'rmse-{0}-{1}'
    fig, ax = plt.subplots()
    position = 1.5
    colors = ['red', 'green', 'purple']
    textures = ['sand', 'silt', 'clay']
    cmap = ListedColormap(colors)
    for i in range(len(models_names)):
        plot_df = all_y[[rmse_texture_col_name.format(c, models_names[i]) for c in texture_cols]]
        plot_df.plot(kind='bar', stacked=True, width=0.15, position=position, ax=ax, alpha=0.7, colormap=cmap)
        plt.title(texture_name + ' rmse per point')
        plt.xlabel('point number')
        plt.ylabel('rmse')
        position += 1
    plt.legend(handles=[mpatches.Patch(color=colors[k], label=textures[k].split("_")[0]) for k in range(3)], bbox_to_anchor=(0.05, 0.96),
               loc='upper left')
    x = np.arange(7) + 0.5
    plt.xticks(x - 1, all_y.index)
    x_offset = -0.05
    y_offset = 0.02
    name_counter = 0
    models_names = ['RF', 'LR', 'CO', 'CI', 'NN']
    down_counter = 0
    start_annotate_val = 14
    stop_annotate_val = 20
    for p in ax.patches:
        if down_counter >= start_annotate_val:
            b = p.get_bbox()
            val = models_names[name_counter % len(models_names)]
            ax.annotate(val, ((b.x0 + b.x1) / 2 + x_offset, b.y1 + y_offset))
            if down_counter == stop_annotate_val:
                name_counter += 1
                down_counter = 0
            else:
                down_counter += 1
        else:
            down_counter += 1
    plt.savefig('results_all_models/images/rmse point-test {0}'.format(texture_name))
    # plt.show()


def get_image_name(index):
    return 'images_cut_n_augment/{0}_0.png'.format(index)


def plot_textures(df_textures, texture_name):
    # print(df_textures)
    fig, ax = plt.subplots()
    position = 1.5
    colors = ['red', 'green', 'purple']
    textures = df_textures.columns
    cmap = ListedColormap(colors)
    plot_df = df_textures[df_textures.columns]
    plot_df.plot(kind='bar', stacked=True, width=0.6, position=position, ax=ax, alpha=0.7, colormap=cmap)
    plt.title(texture_name + ' texture per point')
    plt.xlabel('point number')
    plt.ylabel('percent(%)')
    plt.legend(handles=[mpatches.Patch(color=colors[k], label=textures[k].split('_')[0]) for k in range(3)], bbox_to_anchor=(0.93, 1.07),
               loc='upper left')
    x = np.arange(len(df_textures.index)) + 0.0
    plt.xticks(x - 0.65, df_textures.index)
    plt.savefig('results_all_models/images/test points {0}'.format(texture_name))
    # plt.show()


def create_df_classes(all_y, soil_class_name):
    with open('train nums', 'rb') as f:
        train_nums = pickle.load(f)
    with open('test nums', 'rb') as f:
        test_nums = pickle.load(f)
    cols = ['train set classes', 'count', 'test set classes', 'count']
    train = all_y[soil_class_name].ix[train_nums].value_counts()
    test = all_y[soil_class_name].ix[test_nums].value_counts()
    lst = []
    for i in range(len(train.index)):
        if len(test.index) > i:
            curr = [train.index[i], train.values[i], test.index[i], test.values[i]]
        else:
            curr = [train.index[i], train.values[i], "", ""]
        lst.append(curr)
    res = pd.DataFrame(lst, columns=cols)
    res.to_excel(train.name + '.xlsx', index=False)


def plot_results_model_by_texture_class():
    with open('train nums', 'rb') as f:
        train_nums = list(pickle.load(f))
    with open('test nums', 'rb') as f:
        test_nums = pickle.load(f)
    results_dir = 'results_all_models/'
    models = [RandomForest, LinearReg, Conv, Conv_img, NN]
    cols_res_df = ['target', "model", 'validation R^2', 'validation RMSE']
    for texture_name in ['master sizer', 'hydro meter']:
        with open('texture cols {0}'.format(texture_name), 'rb') as f:
            texture_cols = pickle.load(f)
            results_df = pd.DataFrame(columns=cols_res_df)
            results_df_name = 'results matched to fig-{0}.xlsx'.format(texture_name)
        for model_name in [m.model_name for m in models]:
            prediction_by_point_df = pd.read_excel('res by points all models-{0}.xlsx'.format(texture_name))
            prediction_by_point_df.index += 1
            fig, axes = plt.subplots(3, 1)
            prediction_col = '{0}-{1}'
            rmse_col = 'rmse-{0}-{1}'
            sum_rmse = 0
            for i in range(3):
                curr_y = list(prediction_by_point_df[texture_cols[i]][test_nums].values)
                curr_p = list(prediction_by_point_df[prediction_col.format(texture_cols[i], model_name)][test_nums].values)
                curr_y_train = list(prediction_by_point_df[texture_cols[i]][train_nums].values)
                curr_p_train = list(prediction_by_point_df[prediction_col.format(texture_cols[i], model_name)][train_nums].values)
                min_val_x, max_val_x, min_val_y, max_val_y = min(curr_p + curr_p_train), max(curr_p + curr_p_train), \
                                                             min(curr_y + curr_p), max(curr_y + curr_p)
                axes[i].plot([min_val_x, max_val_x], [min_val_y, max_val_y])
                axes[i].scatter(curr_p, curr_y, color='g')
                axes[i].scatter(curr_p_train, curr_y_train, color='b', marker='+')

                r_squared = calculate_r_square(curr_p, curr_y, (max_val_y + min_val_y) / 2)

                rmse_val = np.mean(np.power(prediction_by_point_df[rmse_col.format(texture_cols[i], model_name)][test_nums], 2))
                sum_rmse += rmse_val
                rmse_val = int(rmse_val * 100) / 100
                res_str = "r squared = {0}\n RMSE = {1}".format(r_squared, rmse_val)

                axes[i].plot([], [], ' ', label=res_str)
                axes[i].set_xlabel(texture_cols[i] + " predicted")
                axes[i].set_ylabel(texture_cols[i] + ' real value')
                axes[i].legend()

                res_dict = {'target': [texture_cols[i]], "model": [model_name],
                            'validation R^2': [r_squared], 'validation RMSE': [rmse_val]}
                row = pd.DataFrame.from_dict(res_dict)
                results_df = pd.concat([results_df, row], sort=True)

            sum_str = 'sum_' + ('h' if texture_name == 'hydro meter' else 'm')
            # sum_rmse = prediction_by_point_df[rmse_col.format(sum_str, model_name)].sum()
            sum_rmse = int(sum_rmse * 100) / 100
            res_dict = {'target': ['sum rmse'], 'texture type': [texture_name], "model": [model_name],
                         'validation R^2': [""], 'validation RMSE': [sum_rmse]}
            row = pd.DataFrame.from_dict(res_dict)
            results_df = pd.concat([results_df, row], sort=True)
            plt.suptitle(model_name)
            plt.savefig(results_dir + '/images/{0} {1}'.format(model_name, texture_name))
        temp = pd.DataFrame()
        for col in cols_res_df:
            temp[col] = results_df[col]
        results_df = temp
        results_df = results_df.sort_values(by=['target', 'model'])
        results_df.to_excel(results_df_name.format(texture_name), index=False)


# main()
plot_results_model_by_texture_class()

