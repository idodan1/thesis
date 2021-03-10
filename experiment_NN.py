import pickle
from GA_NN import *
from class_NN import *
from LinearRegression import *

all_data_file = 'soil_data_2020_all data.xlsx'
all_data_df = pd.read_excel(all_data_file, index_col=0)[2:]
results_df = pd.read_excel('results net.xlsx', index=False)

with open('train_nums_net', 'rb') as f:
    train_nums = pickle.load(f)
with open('val_nums_net', 'rb') as f:
    val_nums = pickle.load(f)
with open('test_nums', 'rb') as f:
    test_nums = pickle.load(f)
with open('cols_for_model', 'rb') as f:
    cols_for_model = pickle.load(f)
with open('texture_master_cols', 'rb') as f:
    texture_master_cols = pickle.load(f)
with open('texture_hydro_cols', 'rb') as f:
    texture_hydro_cols = pickle.load(f)

model = NN
texture_name = 'master sizer'
texture_cols = texture_master_cols if texture_name == 'master sizer' else texture_hydro_cols
feature_len = len(cols_for_model)
num_iter = 20
pop_size = 30
train_df = all_data_df.ix[train_nums]
val_df = all_data_df.ix[val_nums]
test_df = all_data_df.ix[test_nums]
length_penalty = 0.001
max_num_layers = 7
min_num_neurons = 3
max_num_neurons = 10
best_member, best_loss, best_member_net = iterate(model, num_iter, feature_len, pop_size, train_df, test_df,
                                                  texture_cols, cols_for_model, length_penalty, max_num_layers,
                                                  min_num_neurons, max_num_neurons, val_df)

print('min model lost = {0}'.format(best_loss))
print("num of features = {0}".format(sum(best_member)))
print('net architecture = {0}'.format(best_member_net))
print("features = {0}".format([cols_for_model[i] for i in range(len(cols_for_model)) if best_member[i] == 1]))
print()

if str(model) not in results_df['model name'].values or\
        (results_df.loc[results_df['model name'] == str(model)])['texture type'].values[0] != texture_name:
    model_cols = [cols_for_model[i] for i in range(len(cols_for_model)) if best_member[i] == 1]
    row_df = pd.DataFrame([[str(model), best_loss, texture_name, str(model_cols), str(best_member_net)]], columns=results_df.columns)
    results_df = pd.concat([results_df, row_df], ignore_index=True)
else:
    if (results_df.loc[(results_df['model name'] == str(model)) &
                       (results_df['texture type'] == texture_name)])['model loss'].values[0] > best_loss:
        index = list(results_df.loc[results_df['model name'] == str(model)].index)[0]
        model_cols = [cols_for_model[i] for i in range(len(cols_for_model)) if best_member[i] == 1]
        results_df.loc[index, 'model loss'] = best_loss
        results_df.loc[index, 'features'] = str(model_cols)
        results_df.loc[index, 'net architecture'] = str(best_member_net)

results_df.to_excel('results net.xlsx', index=False)
