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
from tensorflow.keras.callbacks import EarlyStopping


class NN:
    def __init__(self, train_x, val_x, test_x, train_y, val_y, test_y):
        self.train_x, self.val_x, self.test_x, self.train_y, self.val_y, self.test_y = train_x, val_x, test_x, train_y,\
                                                                                       val_y, test_y

    def create_model(self, num_of_neurons):
        self.model = tf.keras.models.Sequential()
        self.model.add(Input(shape=len(self.train_x.columns)))

        for i in range(len(num_of_neurons)):
            self.model.add(Dense(num_of_neurons[i]*10, activation=activations.relu))
        self.model.add(Dense(3))

        self.model.compile(
            loss=tf.keras.losses.MeanAbsolutePercentageError(),
            optimizer='adam', metrics=['accuracy']
        )

    def train(self, val_percent=0.7):
        x = [list(rows.values) for index, rows in self.train_x.iterrows()]
        y = [list(rows.values) for index, rows in self.train_y.iterrows()]
        val_x = [list(rows.values) for index, rows in self.val_x.iterrows()]
        val_y = [list(rows.values) for index, rows in self.val_y.iterrows()]
        stop_when_enough = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, restore_best_weights=True)
        self.history = self.model.fit(
            x, y,
            validation_data=(val_x, val_y),
            epochs=500, batch_size=5, verbose=0, callbacks=stop_when_enough
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