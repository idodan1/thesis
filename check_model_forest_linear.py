from MlModels import *

"""
This module examine all the possible subsets of the feature vector with Linear Regression and Random Forest models.
"""


def main():
    results_dir = 'results_all_models/'
    train_df, test_df = get_train_test_df()
    cols_for_model = ['ECaV_2019', 'ECaV_man', 'ECaV_2020', 'ECaV_2018', 'DTM', 'mean_slope', 'NDVI_2_2019',
                      'NDVI_12_2018', 'TIR_2_2020', 'TIR_3_2020', 'max_slope']
    cols_for_res_df = ['total rmse', 'permutation num', 'features']

    for model_type in [RandomForest, LinearReg]:
        for texture_name in ['master sizer', 'hydro meter']:
            texture_cols = get_texture_cols(texture_name)
            res_dir_name = results_dir + model_type.model_name
            res_df_name = res_dir_name + "/res df {0}.xlsx".format(texture_name)
            res_df = pd.DataFrame(columns=cols_for_res_df)

            for i in range(1, 2**len(cols_for_model)):
                if i % 20 == 0:
                    print('iter num: {0}'.format(i))
                num_binary = "{0:b}".format(i)
                current_permutation = list(num_binary)
                current_permutation = ['0']*(len(cols_for_model)-len(current_permutation)) + current_permutation
                current_cols = [cols_for_model[i] for i in range(len(cols_for_model)) if current_permutation[i] == "1"]

                train_x, test_x = train_df[current_cols], test_df[current_cols]
                train_y, test_y = train_df[texture_cols], test_df[texture_cols]

                model = model_type(train_x, test_x, train_y, test_y)
                model.create_model()
                model.train()
                predictions = model.predict_test()
                res_model = calc_rmses(predictions, test_df, texture_cols)

                res_dict = {'total rmse': [res_model.sum()], 'permutation num': [i], 'features': [str(current_cols)]}
                row = pd.DataFrame.from_dict(res_dict)
                res_df = pd.concat([res_df, row], sort=True)

            res_df = res_df.sort_values(by=['total rmse'])
            res_df.to_excel(res_df_name, index=False)


main()







