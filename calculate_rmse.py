from class_NN import *
from RandomForest import *
from LinearRegression import *
from class_CONV import *
from RandomModel import *
import ast


def calculate_rmse(y_pred, y_real):
    return (sum([(y_pred[i] - y_real[i])**2 for i in range(len(y_pred))])/len(y_pred))**0.5


def calculate_regular_model(model_type, train_x, test_x, train_y, test_y):
    model = model_type(train_x, test_x, train_y, test_y)
    model.create_model()
    model.train()
    predictions = model.predict_test()
    predictions_train = model.predict_train()
    return predictions, predictions_train


def get_architecture(texture_name, option):
    df = pd.read_excel('results_all_models/NN/res df {0} division: {1}.xlsx'.format(texture_name, option))
    num_of_neurons_lst = []
    activation_nums_lst = []
    for i in range(5):
        num_of_neurons, activation_nums = ast.literal_eval(df['num of neurons'][i]),\
                                          ast.literal_eval(df['activation nums'][i])
        num_of_neurons_lst.append(num_of_neurons)
        activation_nums_lst.append(activation_nums)
    return num_of_neurons_lst, activation_nums_lst


def calculate_nn(train_x, test_x, train_y, test_y, texture_name, option, num_of_neurons, activation_nums):
    model = NN(train_x, test_x, test_x, train_y, test_y, test_y)
    model.create_model(num_of_neurons, activation_nums)
    model.train()
    predictions = model.predict_test()
    predictions_train = model.predict_train()
    return predictions, predictions_train


def print_weights(model):
    i = 0
    for layer in model.layers:
        # if i == 3:
        print("layer i = {0}".format(i))
        print(layer.get_weights())
        print(len(layer.get_weights()))
        i += 1


def forget_augmentation(test_names):
    res = []
    for name in test_names:
        if name.split('.')[0][-1] == '0':
            res.append(name)
    return res


def calculate_conv(train_x, test_x, train_y, test_y, texture_name, option, train_nums):
    num_of_neurons, activation_nums = get_architecture(texture_name, option)
    model = Conv()
    img_size = 2000
    mini_img_size = 100
    num_of_blocks = 2
    num_of_filters = 32
    num_of_mini_img = img_size // mini_img_size
    numeric_size = len(train_x.columns)
    model.create_model(mini_img_size, numeric_size, num_of_neurons, activation_nums, num_of_blocks, num_of_filters)
    batch_size = 7
    val_set_size = 5
    num_iter = 1
    epochs = 2
    img_location = 'images_cut_n_augment'
    images_names = np.array(glob.glob(img_location + '/*.png'))
    train_names, val_names, test_names = divide_images(images_names, val_set_size, train_nums)
    original_train = np.array(list(train_names) + list(val_names))

    val_data = names_to_arr_num(val_names, train_x)
    val_x_img, val_x_num, val_y = names_to_arr_img(val_names, train_y, val_data, mini_img_size, num_of_mini_img)

    history = []
    for i in range(num_iter):
        s = np.random.choice(range(len(train_names)), batch_size)
        numeric_data = names_to_arr_num(train_names[s], train_x)
        batch_x_img, batch_x_num, batch_y = names_to_arr_img(train_names[s], train_y, numeric_data, mini_img_size,
                                                             num_of_mini_img)
        p = np.random.permutation(len(batch_y))
        batch_x_img, batch_x_num, batch_y = batch_x_img[p], batch_x_num[p], batch_y[p]
        model.train(batch_x_img, batch_x_num, batch_y, val_x_img, val_x_num, val_y, batch_size, epochs)
        if i % 5 == 0:
            print("num of iter = {0}".format(i))
            print("train acc = {0}, val acc = {1}".format(model.history.history['accuracy'][-1], model.history.history['val_accuracy'][-1]))
            print("train loss = {0}, val loss = {1}".format(model.history.history['loss'][-1], model.history.history['val_loss'][-1]))
            print()
        history.append([model.history.history['loss'][-1], model.history.history['val_loss'][-1]])
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
    all_predictions_train = []
    all_y_train = []
    gap_train = []
    original_train = forget_augmentation(original_train)
    for i in range(len(original_train)):
        x, y = create_data(train_names[i], train_y)
        x_mini = create_mini_images(x, mini_img_size)
        numeric_data = np.array(names_to_arr_num([original_train[i]], train_x)[0])
        numeric_data = np.array([numeric_data for _ in range(x_mini.shape[0])])
        predictions, gap = model.predict([x_mini, numeric_data])
        gap_train.append(gap)
        all_predictions_train.append(predictions)
        all_y_train.append(y)
    return all_predictions, all_predictions_train, all_y, all_y_train, gaps_test, gap_train


def main():
    results_dir = 'results_all_models/'
    results_df = pd.read_excel('results_all_models/results all models.xlsx')
    all_data_file = 'soil_data_2020_all data.xlsx'
    all_data_df = pd.read_excel(all_data_file, index_col=0)[2:]
    cols_for_model = ['ECaV_2019', 'ECaV_man', 'ECaV_2020', 'DTM', 'mean_slope']
    models_names = ['NN']
    models = [NN]
    # models = [RandomModel, RandomForest, LinearReg]
    # models_names = ['random model', 'random forest', 'linear regression']

    for option in [' random']:
        for texture_name in ['master sizer', 'hydro meter']:
            for k in range(len(models)):
                model_type = models[k]
                print("currently {0} {1}".format(option, texture_name))
                with open('train nums{0} {1}'.format(option, texture_name), 'rb') as f:
                    train_nums = pickle.load(f)
                with open('test nums{0} {1}'.format(option, texture_name), 'rb') as f:
                    test_nums = pickle.load(f)
                with open('texture cols {0}'.format(texture_name), 'rb') as f:
                    texture_cols = pickle.load(f)

                train_df = all_data_df.ix[train_nums]
                """
                    i lost image 13 so i have to erase 13 from train sets
                """
                if 13 in train_df.index:
                    train_df = train_df.drop([13], axis='index')
                test_df = all_data_df.ix[test_nums]
                if 13 in test_df.index:
                    test_df = test_df.drop([13], axis='index')
                train_x, val_x, test_x = train_df[cols_for_model], test_df[cols_for_model], test_df[cols_for_model]
                train_y, val_y, test_y = train_df[texture_cols], test_df[texture_cols], test_df[texture_cols]
                test_y_lst = [list(rows.values) for index, rows in test_y.iterrows()]
                train_y_lst = [list(rows.values) for index, rows in train_y.iterrows()]

                num_of_neurons_lst, activation_nums_lst = get_architecture(texture_name, option)
                for i in range(len(num_of_neurons_lst)):
                    predictions, predictions_train = calculate_nn(train_x, test_x, train_y, test_y, texture_name, option
                                                                  , num_of_neurons_lst[i], activation_nums_lst[i])
                # predictions, predictions_train = calculate_regular_model(model_type, train_x, test_x, train_y, test_y)
                # predictions, predictions_train, all_y, all_y_train, gaps_test, gap_train = calculate_conv(
                #     train_x, test_x, train_y, test_y, texture_name, option, train_nums)
                # test_y_lst = all_y
                # train_y_lst = all_y_train

                    fig, axes = plt.subplots(3, 1)
                    for i in range(3):
                        curr_y = [test_y_lst[j][i] for j in range(len(test_y_lst))]
                        curr_p = [predictions[j][i] for j in range(len(predictions))]
                        curr_y_train = [train_y_lst[j][i] for j in range(len(train_y_lst))]
                        curr_p_train = [predictions_train[j][i] for j in range(len(predictions_train))]
                        min_val_x, max_val_x, min_val_y, max_val_y = min(curr_p+curr_p_train), max(curr_p+curr_p_train), \
                                                                     min(curr_y+curr_p), max(curr_y+curr_p)
                        axes[i].plot([min_val_x, max_val_x], [min_val_y, max_val_y], label='1:1 line')
                        axes[i].scatter(curr_p, curr_y, color='g', label='test')
                        axes[i].scatter(curr_p_train, curr_y_train, color='b', marker='+', label='train')

                        m, b = np.polyfit(curr_p, curr_y, 1)
                        axes[i].plot([min_val_x, max_val_x], m*np.array([min_val_x, max_val_x])+b, label='regression line')

                        correlation_matrix = np.corrcoef(curr_p, curr_y)
                        correlation_xy = correlation_matrix[0, 1]
                        r_squared = correlation_xy ** 2

                        rmse = calculate_rmse(curr_p, curr_y)
                        res_str = "r squared = {0}\n RMSE = {1}".format(r_squared, rmse)

                        axes[i].plot([], [], ' ',  label=res_str)
                        axes[i].set_xlabel(texture_cols[i] + " predicted")
                        axes[i].set_ylabel(texture_cols[i] + ' real value')
                        div = 'gap statistic' if option == '' else 'random'
                        axes[i].legend()

                        res_dict = {'target': [texture_cols[i]], 'texture type': [texture_name], "model": [models_names[k]],
                                    'features': [str(cols_for_model)], "architecture":
                                        [str(num_of_neurons_lst[i]) + str(activation_nums_lst[i])], 'validation R^2':
                                        [r_squared], 'validation RMSE': [rmse]},
                        row = pd.DataFrame.from_dict(res_dict)
                        results_df = pd.concat([results_df, row], sort=True)

                    plt.suptitle(models_names[k] + "\ndivision: {0}".format(div))
                    plt.savefig(results_dir + '/images/{0} {1}'.format(models_names[k], texture_name))
                # plt.show()
    results_df.to_excel('results_all_models/results all models.xlsx', index=False)


# main()


