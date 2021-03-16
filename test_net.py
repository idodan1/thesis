from class_NN import *
import matplotlib.pyplot as plt


all_data_df, train_nums, val_nums, test_nums, cols_for_model, texture_master_cols,\
texture_hydro_cols, activation_funcs = create_df_n_lists_for_model()

train_df = all_data_df.ix[train_nums]
val_df = all_data_df.ix[val_nums]
test_df = all_data_df.ix[test_nums]
texture_cols = texture_hydro_cols

train_y, val_y, test_y = train_df[texture_cols], val_df[texture_cols], test_df[texture_cols]
cols_for_model = ['ECaH_2019', 'ECaH_2018', 'DTM', 'NDVI_2_2019', 'mzs_diff1820', 'MSaV_man', 'flow_distance']
train_x = train_df[cols_for_model]
val_x = val_df[cols_for_model]
test_x = test_df[cols_for_model]

num_of_neurons = [15, 15, 15, 15]
activation_lst = [1, 2, 1, 1]

net = NN(train_x, val_x, test_x, train_y, val_y, test_y)
net.create_model(num_of_neurons,  activation_lst, activation_funcs)
net.train()
predictions = net.predict_test()
loss = net.calc_loss(predictions)
num_epochs = len(net.history.history['loss'])
print("loss: {0}".format(loss))
print("num of epochs: {0}".format(num_epochs))

plt.plot(range(0, num_epochs), net.history.history['loss'], color='green', label='loss')
plt.plot(range(0, num_epochs), net.history.history['val_loss'], color='blue', label='val loss')
plt.legend()
plt.show()

if input('save?(y)') == 'y':
    net.model.save('best_net_model')










    