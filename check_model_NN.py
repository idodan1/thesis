# from class_NN import *
from MlModels import *
from GA_NN import *


def create_permutation(feature_space):
    res = {}
    is_list = True
    num_of_layers = np.random.randint(feature_space['num_of_layers'][0], feature_space['num_of_layers'][1])
    for k in feature_space:
        if feature_space[k][2] == is_list:
            res[k] = np.random.randint(feature_space[k][0], feature_space[k][1], num_of_layers)
        else:
            res[k] = np.random.randint(feature_space[k][0], feature_space[k][1])
    return res


def main():
    model_type = NN
    model_name = model_type.model_name
    is_list = True
    not_list = False
    feature_space = {"batch_size": [3, 10, not_list], "num_of_neurons": [1, 10, is_list], 'activations':
            [0, 3, is_list], "restore_best_weights": [0, 1, not_list],
                     "patience": [10, 50, not_list], "all_the_same": [0, 1, not_list], 'num_of_layers': [1, 7, not_list]}
    texture_name = 'master sizer'
    # texture_name = 'hydro meter'

    # GA parameters
    pop_size = 2
    num_iter = 1
    leave_val = int(pop_size * 0.2)
    new_pop = True

    train_df, test_df, texture_cols, texture_model_cols = get_train_test_cols(texture_name)
    train_x, test_x = train_df[texture_model_cols], test_df[texture_model_cols]
    train_y, test_y = train_df[texture_cols], test_df[texture_cols]

    best_member, best_fit = iterate(model_type, feature_space, train_x, test_x, train_y, test_y, test_df,
                                    texture_cols, pop_size, create_permutation, num_iter, model_name, new_pop,
                                    leave_val, texture_name)
    print('best member = {0}'.format(best_member))
    print('best fit = {0}'.format(best_fit))


main()

