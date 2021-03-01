from class_NN import *


train_x_df = pd.read_excel("train_x.xlsx", index_col=0).sort_values(by=['sample']).drop(['sample'], axis=1)
test_x_df = pd.read_excel("test_x.xlsx", index_col=0).sort_values(by=['sample']).drop(['sample'], axis=1)
train_y_df = pd.read_excel("train_y.xlsx", index_col=0).sort_values(by=['sample']).drop(['sample', 'cluster'], axis=1)
test_y_df = pd.read_excel("test_y.xlsx", index_col=0).sort_values(by=['sample']).drop(['sample', 'cluster'], axis=1)
train_x = [list(train_x_df[i:i+1].values[0]) for i in range(len(train_x_df))]
train_y = [list(train_y_df[i:i+1].values[0]) for i in range(len(train_y_df))]
test_x = [list(test_x_df[i:i+1].values[0]) for i in range(len(test_x_df))]
test_y = [list(test_y_df[i:i+1].values[0]) for i in range(len(test_y_df))]


p = [1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0]
model = NN(train_x, test_x, train_y, test_y, p)
model.create_model()
model.train()
predictions = model.predict_test()
per_data_point = calc_per_point(train_y, predictions)
loss = calc_loss(train_y, predictions)
# plot_textures(train_y_df, per_data_point)
print(loss)






