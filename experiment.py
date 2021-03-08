from RandomForest import *
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import math
from GA import *
from class_NN import *
from LinearRegression import *

all_data_file = 'soil_data_2020_all data.xlsx'
all_data_df = pd.read_excel(all_data_file, index_col=0)[2:]
results_df = pd.read_excel('results all models.xlsx', index=False)
print()

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

model = NN
num_iter = 20
feature_len = len(cols_for_model)
pop_size = 20
train_df = all_data_df.ix[train_nums]
test_df = all_data_df.ix[test_nums]
length_penalty = 0.001
best_member, best_loss = iterate(model, num_iter, feature_len, pop_size, train_df, test_df, texture_master_cols,
                                 cols_for_model, length_penalty)

print('min model lost = {0}'.format(best_loss))
print("num of features = {0}".format(sum(best_member)))
print("features = {0}".format([cols_for_model[i] for i in range(len(cols_for_model)) if best_member[i] == 1]))
print()

if str(model) not in results_df['model name'].values:
    model_cols = [cols_for_model[i] for i in range(len(cols_for_model)) if best_member[i] == 1]
    row_df = pd.DataFrame([[str(model), best_loss, str(model_cols)]], columns=results_df.columns)
    results_df = pd.concat([results_df, row_df], ignore_index=True)
else:
    if (results_df.loc[results_df['model name'] == str(model)])['model loss'].values[0] > best_loss:
        index = list(results_df.loc[results_df['model name'] == str(model)].index)[0]
        model_cols = [cols_for_model[i] for i in range(len(cols_for_model)) if best_member[i] == 1]
        results_df.loc[index, 'model loss'] = best_loss
        results_df.loc[index, 'features'] = str(model_cols)

results_df.to_excel('results all models.xlsx', index=False)








"""
train_df = all_data_df.ix[train_nums]
test_df = all_data_df.ix[test_nums]
train_y, test_y = train_df[texture_master_cols], test_df[texture_master_cols]

feature_list = [1]*len(cols_for_model)
cols_for_model = [cols_for_model[i] for i in range(len(cols_for_model)) if feature_list[i] == 1]
train_x = train_df[cols_for_model]
test_x = test_df[cols_for_model]
max_depth = 10

model = RandomForest(train_x=train_x, test_x=test_x, train_y=train_y, test_y=test_y)
model.create_model(max_depth=max_depth)
model.train()
predictions = model.predict_test()
loss = model.calc_loss(predictions)
print(loss)
"""