from calculate_rmse import *
import os


def main():
    results_dir = 'results_all_models/'
    all_data_file = 'soil_data_2020_all data.xlsx'
    all_data_df = pd.read_excel(all_data_file, index_col=0)[2:]
    cols_for_model_by_texture = [['ECaV_man', 'ECaV_2020', 'DTM', 'mean_slope', 'NDVI_12_2018'],
                                 ['ECaV_man', 'ECaV_2020', 'DTM', 'NDVI_2_2019']]

    model_name = 'Conv'
    cols_for_res_df = ['total loss', 'texture type', "num of features", 'features',
                       "num of blocks", 'num of filters', 'training by', 'early_stopping', 'num of neurons',
                       'activation nums']
    mini_img_size = 100
    img_size = 2000
    min_num = 5
    max_num = 6
    batch_size = 7
    val_set_size = 1
    epochs = 3
    training_by = 'loss'
    early_stopping = 10
    num_of_mini_img = img_size // mini_img_size
    texture_names = ['master sizer', 'hydro meter']
    for option in [' random']:
        for texture_name in texture_names:
        # for texture_name in ['master sizer']:
            texture_num = texture_names.index(texture_name)
            print("currently {0} {1}".format(option, texture_name))

            res_dir_name = results_dir + model_name
            res_df_name = res_dir_name + "/res df {0} division: {1}.xlsx".format(texture_name, option)
            if os.path.isfile(res_df_name):
                res_df = pd.read_excel(res_df_name)
            else:
                res_df = pd.DataFrame(columns=cols_for_res_df)
            num_of_neurons_lst, activation_nums_lst = get_architecture(texture_name, option, 0, 2)
            for k in range(len(num_of_neurons_lst)):
                num_of_neurons = num_of_neurons_lst[k]
                activation_nums = activation_nums_lst[k]
                cols_for_model = cols_for_model_by_texture[texture_num]
                with open('train nums{0} {1}'.format(option, texture_name), 'rb') as f:
                    train_nums = pickle.load(f)
                with open('test nums{0} {1}'.format(option, texture_name), 'rb') as f:
                    test_nums = pickle.load(f)
                with open('texture cols {0}'.format(texture_name), 'rb') as f:
                    texture_cols = pickle.load(f)

                train_df = all_data_df.ix[train_nums]
                test_df = all_data_df.ix[test_nums]
                train_x, val_x, test_x = train_df[cols_for_model], test_df[cols_for_model], test_df[cols_for_model]
                train_y, val_y, test_y = train_df[texture_cols], test_df[texture_cols], test_df[texture_cols]

                for num_of_blocks in range(min_num, max_num):
                    for num_of_filters_power in range(2, max_num):
                        num_of_filters = 2**num_of_filters_power
                        print('num_of_blocks num: {0}\nnum_of_filters: {1}'.format(num_of_blocks, num_of_filters))
                        numeric_size = len(train_x.columns)
                        model = Conv()

                        model.create_model(mini_img_size, numeric_size, num_of_neurons, activation_nums, num_of_blocks,
                                           num_of_filters)

                        img_location = 'images_cut_n_augment'
                        images_names = np.array(glob.glob(img_location + '/*.png'))
                        train_names, val_names, test_names = divide_images(images_names, val_set_size, train_nums)

                        val_data = names_to_arr_num(val_names, train_x)
                        val_x_img, val_x_num, val_y = names_to_arr_img(val_names, train_y, val_data, mini_img_size,
                                                                       num_of_mini_img)

                        history = []
                        lowest_loss = np.inf
                        best_model = 0
                        counter_early_stopping = 0
                        total_counter = 0
                        while total_counter < 100:
                            total_counter += 1
                            if total_counter % 20 == 0:
                                print('counter = ', total_counter)
                                print('lowest loss = ', lowest_loss)
                            s = np.random.choice(range(len(train_names)), batch_size)
                            numeric_data = names_to_arr_num(train_names[s], train_x)
                            batch_x_img, batch_x_num, batch_y = names_to_arr_img(train_names[s], train_y, numeric_data,
                                                                                 mini_img_size,
                                                                                 num_of_mini_img)
                            p = np.random.permutation(len(batch_y))
                            batch_x_img, batch_x_num, batch_y = batch_x_img[p], batch_x_num[p], batch_y[p]
                            model.train(batch_x_img, batch_x_num, batch_y, val_x_img, val_x_num, val_y, batch_size, epochs)
                            history.append([model.history.history['loss'][-1], model.history.history['val_loss'][-1]])
                            if lowest_loss > model.history.history[training_by][-1]:
                                lowest_loss = model.history.history[training_by][-1]
                                best_model = model
                                counter_early_stopping = 0
                            elif counter_early_stopping == early_stopping:
                                break
                            else:
                                counter_early_stopping += 1

                        model = best_model
                        all_predictions = []
                        gaps_test = []
                        all_y = []
                        test_names = forget_augmentation(test_names)
                        for i in range(len(test_names)):
                            x, y = create_data(test_names[i], test_y)
                            x_mini = create_mini_images(x, mini_img_size)
                            numeric_data = np.array(names_to_arr_num([test_names[i]], test_x)[0])
                            numeric_data = np.array([numeric_data for _ in range(x_mini.shape[0])])
                            predictions, gap = model.predict([x_mini, numeric_data])
                            all_predictions.append(predictions)
                            gaps_test.append(gap)
                            all_y.append(y)

                        loss = model.calc_loss(all_predictions, all_y)
                        print('loss = ', loss)
                        print('total epochs = ', total_counter)
                        print()
                        res_dict = {'total loss': [loss], 'texture type': [texture_name], "num of features":
                            [len(cols_for_model)], 'division': [option], 'features': [str(cols_for_model)], "num of blocks":
                            [str(num_of_blocks)], 'num of filters': [str(num_of_filters)], 'training by': [training_by],
                            'early_stopping': [early_stopping], 'num of neurons': [num_of_neurons],
                            'activation nums': [activation_nums]}
                        row = pd.DataFrame.from_dict(res_dict)
                        res_df = pd.concat([res_df, row], sort=True)

            res_df = res_df.sort_values(by=['total loss'])
            res_df.to_excel(res_df_name, index=False)


main()






