from RandomModel import *
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import math

all_data_file = 'soil_data_2020_all data.xlsx'
all_data_df = pd.read_excel(all_data_file, index_col=0)[2:]
cols_for_model = ['ECaV_2019', 'ECaV_man', 'ECaV_2020', 'ECaV_2018', 'DTM', 'mean_slope', 'NDVI_2_2019', 'NDVI_12_2018']
for option in [' random', '']:
    for texture_name in ['master sizer', 'hydro meter']:
        print("currently {0} {1}".format(option, texture_name))
        with open('train nums{0} {1}'.format(option, texture_name), 'rb') as f:
            train_nums = pickle.load(f)
        with open('test nums{0} {1}'.format(option, texture_name), 'rb') as f:
            test_nums = pickle.load(f)
        with open('texture cols {0}'.format(texture_name), 'rb') as f:
            texture_cols = pickle.load(f)

        train_df = all_data_df.ix[train_nums]
        train_x, train_y = train_df[cols_for_model], train_df[texture_cols]
        test_df = all_data_df.ix[test_nums]
        test_x, test_y = test_df[cols_for_model], test_df[texture_cols]

        num_of_iter = 100000
        history = []
        for i in range(num_of_iter):
            model = RandomModel(train_x=train_x, test_x=test_x, train_y=train_y, test_y=test_y, features_list=[])
            model.create_model()
            model.train()
            predictions = model.predict_test()
            loss = model.calc_loss(predictions)
            history.append(loss)

        mean, std = np.mean(history), np.std(history)
        w = 0.01
        n = math.ceil((max(history) - min(history))/w)
        plt.hist(history, label='mean = {0:.2f}\nstd = {1:.2f}'.format(mean, std), bins=n)
        plt.ylabel('count')
        plt.xlabel('loss')
        division = option if option == ' random' else 'gap statistic'
        plt.title('random model loss histogram {0}\ndivision:{1}'.format(texture_name, division))
        plt.legend()
        plt.savefig('random model loss histogram {0}\n division:{1}'.format(texture_name, division))
        # plt.show()
        plt.clf()






