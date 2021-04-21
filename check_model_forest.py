from RandomForest import *
from LinearRegression import *
from functions import *


def main():
    results_dir = 'results_all_models/'
    all_data_file = 'soil_data_2020_all data.xlsx'
    all_data_df = pd.read_excel(all_data_file, index_col=0)[2:]
    cols_for_model = ['ECaV_2019', 'ECaV_man', 'ECaV_2020', 'ECaV_2018', 'DTM', 'mean_slope', 'NDVI_2_2019', 'NDVI_12_2018']
    option = ' random'
    for model_type in [RandomForest, LinearReg]:
        for texture_name in ['master sizer', 'hydro meter']:
            print("currently {0} {1}".format(model_type, texture_name))
            with open('train nums{0} {1}'.format(option, texture_name), 'rb') as f:
                train_nums = pickle.load(f)
            with open('test nums{0} {1}'.format(option, texture_name), 'rb') as f:
                test_nums = pickle.load(f)
            with open('texture cols {0}'.format(texture_name), 'rb') as f:
                texture_cols = pickle.load(f)

            cols_for_res_df = ['total rmse', 'texture type', "num of features", 'division', 'permutation num', 'features']
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
                train_df = all_data_df.ix[train_nums]
                test_df = all_data_df.ix[test_nums]
                train_x, test_x = train_df[current_cols], test_df[current_cols]
                train_y, test_y = train_df[texture_cols], test_df[texture_cols]

                model = model_type(train_x, test_x, train_y, test_y)
                model.create_model()
                model.train()
                predictions = model.predict_test()
                res_model = calc_rmses(predictions, test_df, texture_cols)

                res_dict = {'total rmse': [res_model[-1]], 'texture type': [texture_name], "num of features": [len(current_cols)],
                            'division': [option], 'permutation num': [i], 'features': [str(current_cols)]}
                row = pd.DataFrame.from_dict(res_dict)
                res_df = pd.concat([res_df, row], sort=True)

            res_df = res_df.sort_values(by=['total rmse'])
            res_df.to_excel(res_df_name, index=False)


main()







