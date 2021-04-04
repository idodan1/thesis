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

with open('train_nums', 'rb') as f:
    train_nums = pickle.load(f)
with open('test_nums', 'rb') as f:
    test_nums = pickle.load(f)
with open('cols_for_model', 'rb') as f:
    cols_for_model = pickle.load(f)
with open('texture cols master sizer', 'rb') as f:
    texture_master_cols = pickle.load(f)
with open('texture cols hydro meter', 'rb') as f:
    texture_hydro_cols = pickle.load(f)

for model in [RandomForest, LinearReg]:
# model = NN
    texture_name = 'hydro meter'
    texture_cols = texture_master_cols if texture_name == 'master sizer' else texture_hydro_cols
    num_iter = 50
    feature_len = len(cols_for_model)
    pop_size = 100
    train_df = all_data_df.ix[train_nums]
    test_df = all_data_df.ix[test_nums]
    length_penalty = 0.001
    best_member, best_loss = iterate(model, num_iter, feature_len, pop_size, train_df, test_df, texture_cols,
                                     cols_for_model, length_penalty)

    print('min model lost = {0}'.format(best_loss))
    print("num of features = {0}".format(sum(best_member)))
    print("features = {0}".format([cols_for_model[i] for i in range(len(cols_for_model)) if best_member[i] == 1]))
    print()

    if str(model) not in results_df['model name'].values or\
            (results_df.loc[results_df['model name'] == str(model)])['texture type'].values[0] != texture_name:
        model_cols = [cols_for_model[i] for i in range(len(cols_for_model)) if best_member[i] == 1]
        row_df = pd.DataFrame([[str(model), best_loss, texture_name, str(model_cols)]], columns=results_df.columns)
        results_df = pd.concat([results_df, row_df], ignore_index=True)
    else:
        if (results_df.loc[(results_df['model name'] == str(model)) &
                           (results_df['texture type'] == texture_name)])['model loss'].values[0] > best_loss:
            index = list(results_df.loc[results_df['model name'] == str(model)].index)[0]
            model_cols = [cols_for_model[i] for i in range(len(cols_for_model)) if best_member[i] == 1]
            results_df.loc[index, 'model loss'] = best_loss
            results_df.loc[index, 'features'] = str(model_cols)

    results_df.to_excel('results all models.xlsx', index=False)





"""
we would like to examine more variables to check range of in the net architecture inside the GA. maybe check range of
length_penalty. get the CNN model ready for tests. think about more details to add to the results df, like length 
penalty or maybe others.  
"""


"""
train_df = all_data_df.ix[train_nums]
test_df = all_data_df.ix[test_nums]
train_y, test_y = train_df[texture cols master sizer], test_df[texture cols master sizer]

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