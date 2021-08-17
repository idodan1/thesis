from MlModels import *
from GA_NN import *
import copy


def create_permutation(feature_space_original):
    res = dict()
    num_of_layers = 0
    feature_space = copy.deepcopy(feature_space_original)
    if 'num_of_layers' in feature_space:
        num_of_layers = np.random.randint(feature_space['num_of_layers'][0], feature_space['num_of_layers'][1])
    for k in feature_space:
        if list in feature_space[k]:
            res[k] = np.random.randint(feature_space[k][0], feature_space[k][1], num_of_layers)
        else:
            res[k] = np.random.randint(feature_space[k][0], feature_space[k][1])
    return res


def main():
    for model_type in [NN, ConvImg, Conv]:
        model_name = model_type.model_name
        feature_space = model_type.feature_space
        for texture_name in ['master sizer', 'hydro meter']:
            # GA parameter
            pop_size = 20
            num_iter = 15
            leave_val = int(pop_size * 0.1)
            threshold_mutation = 0.1

            train_df, test_df, texture_cols, texture_model_cols = get_train_test_cols(texture_name)
            train_x, test_x = train_df[texture_model_cols], test_df[texture_model_cols]
            train_y, test_y = train_df[texture_cols], test_df[texture_cols]

            best_member, best_fit = iterate(model_type, feature_space, train_x, test_x, train_y, test_y, test_df,
                                            texture_cols, pop_size, create_permutation, num_iter, model_name,
                                            leave_val, texture_name, threshold_mutation)
            print('best member = {0}'.format(best_member))
            print('best fit = {0}'.format(best_fit))


main()





