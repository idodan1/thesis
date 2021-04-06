from calculate_rmse import *
import os


def create_permutation(feature_space):
    res_dic = {}
    for f in feature_space:
        res_dic[f] = np.random.randint(feature_space[f][0], feature_space[f][1])
    return res_dic


def main():
    results_dir = 'results_all_models/'
    all_data_file = 'soil_data_2020_all data.xlsx'
    all_data_df = pd.read_excel(all_data_file, index_col=0)[2:]
    cols_for_model_by_texture = [['ECaV_man', 'ECaV_2020', 'DTM', 'mean_slope', 'NDVI_12_2018'],
                                 ['ECaV_man', 'ECaV_2020', 'DTM', 'NDVI_2_2019']]

    model_name = 'super model'
    cols_for_res_df = ['total loss', 'texture type', 'epochs', 'over_fitting', 'val_set_size', 'batch_size',
                       "num of blocks", 'num of filters', 'training by', 'early_stopping', 'num of neurons',
                       'activation nums']
    mini_img_size = 100
    img_size = 2000
    num_of_mini_img = img_size // mini_img_size
    pop_size = 10
    feature_space = {"batch_size": [2, 10], "training_by": [0, 2], "val_set_size": [1, 7], "epochs": [2, 10],
                     'early_stopping': [5, 15], 'over_fitting': [0, 1]}
    pop = [create_permutation(feature_space) for _ in range(pop_size)]
    training_by_lst = ['loss', 'val_loss']
    # texture_names = ['master sizer', 'hydro meter']
    texture_names = ['master sizer']
    option = ' random'
    for texture_name in texture_names:
        texture_num = texture_names.index(texture_name)
        print("currently {0} {1}".format(option, texture_name))

        res_dir_name = results_dir + model_name
        res_df_name = res_dir_name + "/res df {0}.xlsx".format(texture_name)
        if os.path.isfile(res_df_name):
            res_df = pd.read_excel(res_df_name)
        else:
            res_df = pd.DataFrame(columns=cols_for_res_df)
        num_of_neurons_lst, activation_nums_lst = get_architecture(texture_name, option, 0, 2)
        num_of_neurons, activation_nums = num_of_neurons_lst[0], activation_nums_lst[1]
        num_of_blocks_lst, num_of_filters_lst = get_architecture_img(texture_name, option, 0, 1)
        num_of_blocks, num_of_filters = num_of_blocks_lst[0], num_of_filters_lst[0]

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
        numeric_size = len(train_x.columns)

        for member in pop:
            model = Conv()
            model.create_model(mini_img_size, numeric_size, num_of_neurons, activation_nums, num_of_blocks,
                               num_of_filters)

            img_location = 'images_cut_n_augment'
            images_names = np.array(glob.glob(img_location + '/*.png'))
            training_by = training_by_lst[member['training_by']]
            val_set_size = member['val_set_size'] if training_by == 'val_loss' else 1
            train_names, val_names, test_names = divide_images(images_names, val_set_size, train_nums)

            val_data = names_to_arr_num(val_names, train_x)
            val_x_img, val_x_num, val_y = names_to_arr_img(val_names, train_y, val_data, mini_img_size,
                                                           num_of_mini_img)

            history = []
            lowest_loss = np.inf
            counter_early_stopping = 0
            total_counter = 0
            while total_counter < 100:
                total_counter += 1
                if total_counter % 20 == 0:
                    print('counter = ', total_counter)
                    print('lowest loss = ', lowest_loss)
                s = np.random.choice(range(len(train_names)), member["batch_size"])
                numeric_data = names_to_arr_num(train_names[s], train_x)
                batch_x_img, batch_x_num, batch_y = names_to_arr_img(train_names[s], train_y, numeric_data,
                                                                     mini_img_size,
                                                                     num_of_mini_img)
                perm = np.random.permutation(len(batch_y))
                batch_x_img, batch_x_num, batch_y = batch_x_img[perm], batch_x_num[perm], batch_y[perm]
                model.train(batch_x_img, batch_x_num, batch_y, val_x_img, val_x_num, val_y, member["batch_size"],
                            member["epochs"])
                history.append([model.history.history['loss'][-1], model.history.history['val_loss'][-1]])
                if lowest_loss > model.history.history[training_by][-1]:
                    lowest_loss = model.history.history[training_by][-1]
                    model.model.save('models/best')
                    counter_early_stopping = 0
                elif counter_early_stopping == member["early_stopping"]:
                    break
                else:
                    counter_early_stopping += 1

            model = tf.keras.models.load_model('models/best')
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
                'early_stopping': [member["early_stopping"]], 'num of neurons': [num_of_neurons],
                'activation nums': [activation_nums], 'epochs': [member["epochs"]], 'over_fitting': [member['over_fitting']],
                'val_set_size': [member['val_set_size']], 'batch_size': [member['batch_size']], }
            row = pd.DataFrame.from_dict(res_dict)
            res_df = pd.concat([res_df, row], sort=True)

    res_df = res_df.sort_values(by=['total loss'])
    res_df.to_excel(res_df_name, index=False)


main()




"""
- option 1: try to improve each net by itself (img net and numeric net), and then unite them together for one super
  model. this requires saving each model and then taking the weights from them for the super model.
- option 2: take the parameters from the winners in each net type and work on this one to make it the best net ever and
  and then find her RMSE. 
"""


"""
for tomorrow: add the option of over_fitting. check as many different options as possible,
start working on the presentation for ofer while doing this. later we can think of different things to cmpre
but this is last important. if we finish we can start writing things.  
"""
