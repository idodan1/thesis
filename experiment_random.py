from RandomModel import *
from functions import *
import matplotlib.pyplot as plt
import math

"""
In this module we examine the random model, num_of_iter specifies how many random models will be created. The results 
is a histogram of the RMSE of the different classes. Target can be one or more columns.
"""


def main():
    for texture_name in ['master sizer', 'hydro meter']:
        train_df, test_df, target_cols, model_cols = get_train_test_cols(texture_name)
        train_y, test_y = train_df[target_cols], test_df[target_cols]

        num_of_iter = 100000
        history = []
        for _ in range(num_of_iter):
            model = RandomModel(train_y=train_y, test_y=test_y)
            model.create_model()
            predictions = model.predict_test()
            res_model = []
            for i in range(len(target_cols)):
                predictions_class = np.array([predictions[k][i] for k in range(len(predictions))])
                y_class = np.array(test_df[target_cols[i]].values)
                rmse_class = calculate_rmse(y_class, predictions_class)
                res_model.append(rmse_class)

            res_model.append(sum(res_model))
            history.append(res_model)

        history = sorted(history, key=lambda x: x[-1])
        target_cols.append('sum\n{0}'.format(texture_name))
        for i in range(len(target_cols)):
            current_col = [history[k][i] for k in range(len(history))]
            current_col = sorted(current_col)
            mean, std = np.mean(current_col), np.std(current_col)
            w = 0.01
            n = math.ceil((max(current_col) - min(current_col)) / w)
            plt.hist(current_col, label='mean = {0:.2f}\nstd = {1:.2f}\n5% values = {2:.2f}'.
                     format(mean, std, current_col[int(len(current_col)*0.05)]), bins=n)
            plt.ylabel('count')
            plt.xlabel('loss')
            plt.title('Random model RMSE histogram - {0}'.format(target_cols[i]))
            plt.legend()
            plt.savefig(
                'results_all_models/random model images/random model loss histogram {0}'.format(target_cols[i]))
            # plt.show()
            plt.clf()


if __name__ == "__main__":
    main()






