import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import image
from PIL import Image, ImageOps
from skimage import io
import colorsys
from copy import deepcopy
import glob
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import random
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dropout, Flatten, Input, Dense, concatenate
from tensorflow.keras import activations, Model
import tensorflow as tf
from functions import *
from tensorflow.keras.callbacks import EarlyStopping


class Conv:
    # def __init__(self, train_x, test_x, train_y, test_y, features_list):
    #     self.train_x, self.test_x, self.train_y, self.test_y = train_x, test_x, train_y, test_y

    def create_model(self, image_size, numeric_size, num_of_neurons, activation_nums, num_of_blocks, num_of_filters):
        # create net for images
        def add_conv_block(input_layer, num_filters):
            _x = Conv2D(num_filters, 3, activation='relu', padding='same')(input_layer)
            _x = BatchNormalization()(_x)
            _x = Conv2D(num_filters, 3, activation='relu')(_x)
            _x = MaxPooling2D(pool_size=2)(_x)
            _x = Dropout(0.5)(_x)
            return _x

        input_img = Input(shape=(image_size, image_size, 3))
        x = add_conv_block(input_img, num_of_filters)
        for i in range(1, num_of_blocks):
            num_of_filters *= 2
            x = add_conv_block(x, num_of_filters)
        x = Flatten()(x)
        x = Dense(50)(x)
        x = Model(inputs=input_img, outputs=x)

        # create net for numeric data
        neurons_constant = numeric_size
        activation_funcs = [activations.sigmoid, activations.relu, activations.linear]
        input_num = Input(shape=numeric_size)
        y = Dense(num_of_neurons[0] * neurons_constant, activation=activation_funcs[activation_nums[0]],
                  kernel_initializer='random_uniform', bias_initializer='random_uniform')(input_num)
        for i in range(1, len(num_of_neurons)):
            y = Dense(num_of_neurons[i]*neurons_constant, activation=activation_funcs[activation_nums[i]],
                  kernel_initializer='random_uniform', bias_initializer='random_uniform')(input_num)
        y = Model(inputs=input_num, outputs=y)

        combined = concatenate([x.output, y.output])
        z = Dense(numeric_size, activation='relu',
                  kernel_initializer='random_uniform', bias_initializer='random_uniform')(combined)
        z = Dense(3, activation='linear',
                  kernel_initializer='random_uniform', bias_initializer='random_uniform')(z)

        # z = Dense(numeric_size, activation='relu',
        #           kernel_initializer='random_uniform', bias_initializer='random_uniform')(y.output)
        # z = Dense(3, activation='linear',
        #           kernel_initializer='random_uniform', bias_initializer='random_uniform')(z)

        # self.model = Model(inputs=[input_num], outputs=z)
        self.model = Model(inputs=[input_img, input_num], outputs=z)


        self.model.compile(
            loss=tf.keras.losses.MeanAbsolutePercentageError(),
            optimizer='adam', metrics=['accuracy']
        )

    def create_model_only_image(self, image_size, num_of_blocks, num_of_filters):
        # create net for images
        def add_conv_block(input_layer, num_filters):
            _x = Conv2D(num_filters, 3, activation='relu', padding='same')(input_layer)
            _x = BatchNormalization()(_x)
            # _x = Conv2D(num_filters, 3, activation='relu')(_x)
            _x = Conv2D(num_filters, 3, activation='relu', padding='same')(_x)
            _x = MaxPooling2D(pool_size=2)(_x)
            _x = Dropout(0.5)(_x)
            return _x

        input_img = Input(shape=(image_size, image_size, 3))
        x = add_conv_block(input_img, num_of_filters)
        for i in range(1, num_of_blocks):
            num_of_filters *= 2
            x = add_conv_block(x, num_of_filters)
        x = Flatten()(x)
        x = Dense(50)(x)
        x = Model(inputs=input_img, outputs=x)

        z = Dense(50, activation='relu',
                  kernel_initializer='random_uniform', bias_initializer='random_uniform')(x.output)
        z = Dense(3, activation='linear',
                  kernel_initializer='random_uniform', bias_initializer='random_uniform')(z)
        self.model = Model(inputs=[input_img], outputs=z)


        self.model.compile(
            loss=tf.keras.losses.MeanAbsolutePercentageError(),
            optimizer='adam', metrics=['accuracy']
        )


    def train(self, x_train_img, x_train_num, y_train, x_val_img, x_val_num, y_val, batch_size, epochs):
        stop_when_enough = EarlyStopping(monitor='loss', min_delta=0, patience=5, restore_best_weights=True)
        self.history = self.model.fit(
            x=[x_train_img, x_train_num], y=y_train,
            validation_data=([x_val_img, x_val_num], y_val),
            epochs=epochs, batch_size=batch_size, verbose=0, callbacks=stop_when_enough
            )

    def train_only_image(self, x_train_img, y_train, x_val_img, y_val, batch_size, epochs):
        stop_when_enough = EarlyStopping(monitor='loss', min_delta=0, patience=5, restore_best_weights=True)
        self.history = self.model.fit(
            x=x_train_img, y=y_train,
            validation_data=(x_val_img, y_val),
            epochs=epochs, batch_size=batch_size, verbose=0, callbacks=stop_when_enough
            )

    def predict(self, x):
        predictions = self.model.predict([x])
        gap = [(np.max(predictions[:, i]) - np.min(predictions[:, i])) for i in range(3)]
        prediction = [np.mean(predictions[:, 0]), np.mean(predictions[:, 1]), np.mean(predictions[:, 2])]
        total = sum(prediction) / 100
        norm = [p / total for p in prediction]
        return norm, gap

    def predict_img(self, x):
        predictions = self.model.predict([x])
        gap = [(np.max(predictions[:, i]) - np.min(predictions[:, i])) for i in range(3)]
        prediction = [np.mean(predictions[:, 0]), np.mean(predictions[:, 1]), np.mean(predictions[:, 2])]
        total = sum(prediction) / 100
        norm = [p / total for p in prediction]
        return norm, gap

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = tf.keras.load_model(path)

    def calc_loss(self, predictions, test_y):
        # test_y = [list(rows.values) for index, rows in test_y.iterrows()]
        return calc_loss(test_y, predictions)


def divide_images(images_names, val_set_size, train_num):
    val_num = np.random.choice(list(train_num), val_set_size)
    train_num = list(set(train_num) - set(val_num))
    train_names = []
    val_names = []
    test_names = []
    for img in images_names:
        img_num = int((img.split("/")[1]).split("_")[0])
        if img_num in train_num:
            train_names.append(img)
        elif img_num in val_num:
            val_names.append(img)
        else:
            test_names.append(img)
    return np.array(train_names), np.array(val_names), np.array(test_names)


def names_to_arr_img(sample_names, df_texture, numeric_data, mini_img_size, num_of_mini_images):
    x_mini, x_num_data, y_res = [], [], []
    for i in range(len(sample_names)):
        x, y = create_data(sample_names[i], df_texture)
        x_mini.extend(create_mini_images(x, mini_img_size))
        x_num_data.extend([numeric_data[i] for _ in range(num_of_mini_images)])
        y_res.extend([y for _ in range(num_of_mini_images)])
    return np.array(x_mini), np.array(x_num_data), np.array(y_res)


def names_to_arr_img_only_img(sample_names, df_texture, mini_img_size, num_of_mini_images):
    x_mini, y_res = [], []
    for i in range(len(sample_names)):
        x, y = create_data(sample_names[i], df_texture)
        x_mini.extend(create_mini_images(x, mini_img_size))
        y_res.extend([y for _ in range(num_of_mini_images)])
    return np.array(x_mini), np.array(y_res)


def names_to_arr_num(sample_names, df_num):
    res = []
    for ad in sample_names:
        label = (ad.split("/")[-1]).split("_")[0]  # cut the label from location
        line = df_num.ix[int(label)]
        res.append(list(line.values))
    return np.array(res)


def main():
    train_x_num = pd.read_excel("train_x.xlsx", index_col=0).sort_values(by=['sample'])
    test_x_num = pd.read_excel("test_x.xlsx", index_col=0).sort_values(by=['sample'])

    location = 'images_cut_n_augment'
    models_path = 'models/train_model'
    df_texture = pd.read_excel('texture df.xlsx')
    img_size = 2000
    mini_img_size = 100
    num_of_mini_img = img_size//mini_img_size
    val_set_size = 5
    numeric_input_len = 14
    images_names = np.array(glob.glob(location + '/*.png'))
    train_names, val_names, test_names = divide_images(images_names, val_set_size)

    model = Conv()
    model.create_model(mini_img_size, numeric_input_len)
    batch_size = 5
    num_iter = 3
    epochs = 3

    val_data = names_to_arr_num(val_names, train_x_num)
    val_x_img, val_x_num, val_y = names_to_arr_img(val_names, df_texture, val_data, mini_img_size, num_of_mini_img)

    history = []
    for i in range(num_iter):
        s = np.random.choice(range(len(train_names)), batch_size)
        numeric_data = names_to_arr_num(train_names[s], train_x_num)
        batch_x_img, batch_x_num, batch_y = names_to_arr_img(train_names[s], df_texture, numeric_data, mini_img_size, num_of_mini_img)
        p = np.random.permutation(len(batch_y))
        batch_x_img, batch_x_num, batch_y = batch_x_img[p], batch_x_num[p], batch_y[p]

        model.train(batch_x_img, batch_x_num, batch_y, val_x_img, val_x_num, val_y, batch_size, epochs)

        if i % 5 == 0:
            print("num of iter = {0}".format(i))
            print("train acc = {0}, val acc = {1}".format(model.history.history['accuracy'][-1], model.history.history['val_accuracy'][-1]))
            print("train loss = {0}, val loss = {1}".format(model.history.history['loss'][-1], model.history.history['val_loss'][-1]))
            print()
        history.append([model.history.history['loss'][-1], model.history.history['val_loss'][-1]])

    # model.save_model(models_path)

    # plot loss over iterations
    fig, axes = plt.subplots(2, 1)
    iter_range = list(range(num_iter))
    train = [item[0] for item in history]
    val = [item[1] for item in history]
    axes[0].plot(iter_range, train)
    axes[1].plot(iter_range, val)
    axes[0].set_title('train loss')
    axes[1].set_title('validation loss')
    axes[0].set_ylabel('loss')
    axes[1].set_ylabel('loss')
    axes[0].set_xlabel('iter')
    axes[1].set_xlabel('iter')
    # plt.show()

    # test data
    loss = 0
    all_predictions = []
    all_y = []
    print(len(test_names))
    for i in range(len(test_names)):
        x, y = create_data(test_names[i], df_texture)
        x_mini = create_mini_images(x, mini_img_size)
        numeric_data = np.array(names_to_arr_num([test_names[i]], test_x_num)[0])
        numeric_data = np.array([numeric_data for _ in range(x_mini.shape[0])])
        predictions = model.predict([x_mini, numeric_data])
        all_predictions.append(predictions)
        all_y.append(y)
        # p_loss = calc_loss(predictions, y)
        # loss += p_loss
    loss = calc_loss(all_y, all_predictions)
    print(loss)

# main()



"""
the important thing is to show that with images is better, start from the beginning, check that every step is working
properly, print everything along the way and see that we get reasonable results there. we have to ge results that are 
at least as good as the others. one of the classes is getting 0 every time, check it. test names len is 40, check it.
test size should be 6, we can plot the range of values mini images get so we can show that it is not too big.
sort this class first, get some kind of a result to show and start preparing to the meeting with ofer
"""