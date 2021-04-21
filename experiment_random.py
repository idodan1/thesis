from RandomModel import *
from functions import *
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import math

if __name__ == "__main__":
    all_data_file = 'soil_data_2020_all data.xlsx'
    all_data_df = pd.read_excel(all_data_file, index_col=0)[2:]
    cols_for_model = ['ECaV_2019', 'ECaV_man', 'ECaV_2020', 'ECaV_2018', 'DTM', 'mean_slope', 'NDVI_2_2019', 'NDVI_12_2018']
    for texture_name in ['master sizer', 'hydro meter']:
        train_df, test_df, texture_cols = get_train_test_cols(texture_name)

        train_x, train_y = train_df[cols_for_model], train_df[texture_cols]
        test_x, test_y = test_df[cols_for_model], test_df[texture_cols]

        num_of_iter = 100000
        history = []
        for _ in range(num_of_iter):
            model = RandomModel(train_x=train_x, test_x=test_x, train_y=train_y, test_y=test_y)
            model.create_model()
            model.train()
            predictions = model.predict_test()
            res_model = []
            for i in range(len(texture_cols)):
                predictions_class = np.array([predictions[k][i] for k in range(len(predictions))])
                y_class = np.array(test_df[texture_cols[i]].values)
                rmse_class = calculate_rmse(y_class, predictions_class)
                res_model.append(rmse_class)

            res_model.append(sum(res_model))
            history.append(res_model)

        texture_cols.append('sum {0}'.format(texture_name))
        for i in range(len(texture_cols)):
            current_col = [history[k][i] for k in range(len(history))]
            mean, std = np.mean(current_col), np.std(current_col)
            w = 0.01
            n = math.ceil((max(current_col) - min(current_col))/w)
            plt.hist(current_col, label='mean = {0:.2f}\nstd = {1:.2f}'.format(mean, std), bins=n)
            plt.ylabel('count')
            plt.xlabel('loss')
            plt.title('random model loss histogram {0}'.format(texture_cols[i]))
            plt.legend()
            plt.savefig('results_all_models/random model images/random model loss histogram {0}'.format(texture_cols[i]))
            # plt.show()
            plt.clf()






