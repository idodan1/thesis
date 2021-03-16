from GA_NN import *
from class_NN import *
from functions import *


all_data_df, train_nums, val_nums, test_nums, cols_for_model, texture_master_cols,\
texture_hydro_cols, activation_funcs = create_df_n_lists_for_model()

model = NN
texture_name = 'hydro meter'
texture_cols = texture_master_cols if texture_name == 'master sizer' else texture_hydro_cols
feature_len = len(cols_for_model)
num_iter = 10
pop_size = 10
train_df = all_data_df.ix[train_nums]
val_df = all_data_df.ix[val_nums]
test_df = all_data_df.ix[test_nums]
length_penalty = 0.005
max_num_layers = 5
min_num_neurons = 2
max_num_neurons = 7
best_member, best_loss, best_member_net, best_activation = iterate(model, num_iter, feature_len, pop_size, train_df,
                                                                   test_df, texture_cols, cols_for_model, length_penalty
                                                                   , max_num_layers, min_num_neurons, max_num_neurons,
                                                                   val_df, activation_funcs)

print('min model lost = {0}'.format(best_loss))
print("num of features = {0}".format(sum(best_member)))
print('net architecture = {0}'.format(best_member_net))
print('net activations = {0}'.format([activation_funcs[k] for k in best_activation]))
print("features = {0}".format([cols_for_model[i] for i in range(len(cols_for_model)) if best_member[i] == 1]))
print()

write_results(model, cols_for_model, best_loss, best_member, best_member_net, best_activation, texture_name,
              activation_funcs)
