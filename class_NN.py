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
from tensorflow.keras.layers import Dropout, Flatten, Input, Dense
from tensorflow.keras import activations
import tensorflow as tf
from functions import *


class NN:
    def __init__(self, train_x, test_x, train_y, test_y):
        self.train_x, self.test_x, self.train_y, self.test_y = train_x, test_x, train_y, test_y

    def create_model(self):
        self.model = tf.keras.models.Sequential()
        self.model.add(Input(shape=len(self.train_x.columns)))

        num_of_neurons = 200
        self.model.add(Dense(num_of_neurons, activation=activations.relu))
        self.model.add(Dense(num_of_neurons / 2, activation=activations.relu))
        self.model.add(Dense(num_of_neurons / 4, activation=activations.relu))
        self.model.add(Dense(num_of_neurons / 5, activation=activations.relu))
        self.model.add(Dense(num_of_neurons / 10, activation=activations.relu))
        self.model.add(Dense(3))

        self.model.compile(
            # loss='mae',
            loss=tf.keras.losses.MeanAbsolutePercentageError(),
            optimizer='adam', metrics=['accuracy']
        )

    def train(self, val_percent=0.7):
        x = [list(rows.values) for index, rows in self.train_x.iterrows()]
        y = [list(rows.values) for index, rows in self.train_y.iterrows()]
        val_index = int(len(x) * 0.7)
        self.history = self.model.fit(
            x[:val_index], y[:val_index],
            validation_data=(x[val_index:], y[val_index:]),
            epochs=200, batch_size=2, verbose=0
        )

    def predict(self, x):
        predictions = self.model.predict([x])[0]
        total = sum(predictions) / 100
        return [p / total for p in predictions]

    def predict_test(self):
        test_x = [list(rows.values) for index, rows in self.test_x.iterrows()]
        return [self.predict(x) for x in test_x]

    def calc_loss(self, predictions):
        test_y = [list(rows.values) for index, rows in self.test_y.iterrows()]
        return calc_loss(test_y, predictions)