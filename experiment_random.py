from RandomModel import *
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import math

all_data_file = 'soil_data_2020_all data.xlsx'
all_data_df = pd.read_excel(all_data_file, index_col=0)[2:]
with open('train_nums', 'rb') as f:
    train_nums = pickle.load(f)
with open('test_nums', 'rb') as f:
    test_nums = pickle.load(f)
with open('cols_for_model', 'rb') as f:
    cols_for_model = pickle.load(f)
with open('texture_master_cols', 'rb') as f:
    texture_master_cols = pickle.load(f)
with open('texture_hydro_cols', 'rb') as f:
    texture_hydro_cols = pickle.load(f)


train_df = all_data_df.ix[train_nums]
train_x, train_y = train_df[cols_for_model], train_df[texture_master_cols]
test_df = all_data_df.ix[test_nums]
test_x, test_y = test_df[cols_for_model], test_df[texture_master_cols]

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
plt.title('random model loss histogram')
plt.legend()
plt.show()
