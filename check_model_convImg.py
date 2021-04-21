from class_conv_img import *
from GA_NN import *
import copy


def create_permutation(feature_space_original):
    res = dict()
    feature_space = copy.deepcopy(feature_space_original)
    res['texture_type'] = feature_space['texture_type']
    del feature_space['texture_type']
    for k in feature_space:
        res[k] = np.random.randint(feature_space[k][0], feature_space[k][1])
    return res


def main():
    model_type = Conv_img
    model_name = model_type.model_name
    texture_name = 'master sizer'
    # texture_name = 'hydro meter'
    feature_space = {"batch_size": [3, 10], "val_set_size": [1, 5], 'training_by': [0, 2],
                     'early_stopping': [1, 10], "monitor": [0, 2], 'num_of_blocks':
                         [1, 6], 'num_of_filters_power': [1, 4], 'epochs': [1, 7], 'num_neurons': [1, 10],
                     'mini_image_size_index': [0, 3], 'activation': [0, 3], 'texture_type': texture_name}

    # GA parameters
    pop_size = 60
    num_iter = 10
    leave_val = int(pop_size * 0.2)
    new_pop = True

    train_df, test_df, texture_cols, texture_model_cols = get_train_test_cols(texture_name)
    train_x, val_x, test_x = train_df[texture_model_cols], test_df[texture_model_cols], test_df[texture_model_cols]
    train_y, val_y, test_y = train_df[texture_cols], test_df[texture_cols], test_df[texture_cols]

    best_member, best_fit = iterate(model_type, feature_space, train_x, val_x, test_x, train_y, val_y, test_y, test_df,
                                    texture_cols, pop_size, create_permutation, num_iter, model_name, new_pop,
                                    leave_val, texture_name)
    print('best member = {0}'.format(best_member))
    print('best fit = {0}'.format(best_fit))


main()

