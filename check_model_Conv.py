from class_CONV import *
from GA_NN import *
import copy
import ast


def get_best(net_type, texture_type):
    df = pd.read_excel('results_all_models/{0}/top5-{1}.xlsx'.format(net_type, texture_type))
    features = ast.literal_eval(df['features'][1])
    return features


def create_permutation(feature_space_original):
    res = dict()
    is_list = True

    # best_num, best_img = get_best('NN', feature_space_original['texture_type']), get_best('conv_img',
    #                                                                                       feature_space_original['texture_type'])
    feature_space = copy.deepcopy(feature_space_original)
    res['texture_type'] = feature_space['texture_type']
    del feature_space['texture_type']
    num_of_layers = np.random.randint(feature_space['num_of_layers'][0], feature_space['num_of_layers'][1])
    # org_features = ['num_of_layers',  "all_the_same", 'activations', 'mini_image_size_index', "num_of_neurons",
    #                 "patience", 'num_of_blocks', 'num_of_filters_power', 'activation', "restore_best_weights",
    #                 'num_neurons']
    # for key in best_num:
    #     if key in org_features:
    #         res[key] = best_num[key]
    # for key in best_img:
    #     if key in org_features:
    #         res[key] = best_img[key]
    # for k in feature_space:
        # if k not in org_features:
        #     res[k] = np.random.randint(feature_space[k][0], feature_space[k][1])
    for k in feature_space:
        if feature_space[k][2] == is_list:
            res[k] = np.random.randint(feature_space[k][0], feature_space[k][1], num_of_layers)
        else:
            res[k] = np.random.randint(feature_space[k][0], feature_space[k][1])

    return res


def main():
    is_list = True
    not_list = False
    model_type = Conv
    model_name = model_type.model_name
    texture_name = 'master sizer'
    # texture_name = 'hydro meter'
    feature_space = {"batch_size": [3, 10, not_list], "val_set_size": [1, 5, not_list], 'training_by': [0, 2, not_list],
                     'early_stopping': [1, 10, not_list], "monitor": [0, 2, not_list], 'epochs': [1, 7, not_list],
                     'texture_type': texture_name, 'num_of_layers': [1, 7, not_list],  "all_the_same": [0, 1, not_list], 'activations': [0, 3, is_list], 'mini_image_size_index': [0, 3, not_list], "num_of_neurons": [1, 10, is_list],
                    "patience": [10, 50, not_list], 'num_of_blocks': [1, 4, not_list], 'num_of_filters_power': [1, 4, not_list], 'activation': [0, 3, not_list], "restore_best_weights": [0, 1, not_list],
                    'num_neurons': [1, 10, not_list]}

    # GA parameters
    pop_size = 50
    num_iter = 3
    leave_val = int(pop_size * 0.1)
    new_pop = False

    train_df, test_df, texture_cols, texture_model_cols = get_train_test_cols(texture_name)
    train_x, val_x, test_x = train_df[texture_model_cols], test_df[texture_model_cols], test_df[texture_model_cols]
    train_y, val_y, test_y = train_df[texture_cols], test_df[texture_cols], test_df[texture_cols]

    best_member, best_fit = iterate(model_type, feature_space, train_x, val_x, test_x, train_y, val_y, test_y, test_df,
                                    texture_cols, pop_size, create_permutation, num_iter, model_name, new_pop,
                                    leave_val, texture_name)
    print('best member = {0}'.format(best_member))
    print('best fit = {0}'.format(best_fit))

main()

