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
    def __init__(self, train_x, test_x, train_y, test_y, features_list):
        self.train_x, self.test_x, self.train_y, self.test_y = select_features(train_x, features_list), select_features(
            test_x, features_list), train_y, test_y

    def create_model(self):
        self.model = tf.keras.models.Sequential()
        self.model.add(Input(shape=len(self.train_x[0])))

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
        val_index = int(len(self.train_x) * 0.7)
        self.history = self.model.fit(
            self.train_x[:val_index], self.train_y[:val_index],
            validation_data=(self.train_x[val_index:], self.train_y[val_index:]),
            ##### check this don't want to touch the test before validation
            epochs=200, batch_size=2, verbose=0
        )

    def predict(self, x):
        predictions = self.model.predict([x])[0]
        total = sum(predictions) / 100
        return [p / total for p in predictions]

    def predict_test(self):
        return [self.predict(x) for x in self.train_x]


