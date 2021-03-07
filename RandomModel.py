import numpy as np
from functions import *


class RandomModel:
    def __init__(self, train_x, test_x, train_y, test_y, features_list):
        self.train_y, self.test_y = train_y, test_y

    def create_model(self):
        self.cols = [self.train_y[col].values for col in self.train_y.columns]
        self.means, self.std = [np.mean(self.cols[i]) for i in range(len(self.cols))], [np.std(self.cols[i]) for i in
                                                                                        range(len(self.cols))]

    def train(self):
        return

    def predict(self, x):
        predictions = [np.random.normal(self.means[i], self.std[i]) for i in range(len(self.means))]
        total = sum(predictions)/100
        return [p/total for p in predictions]

    def predict_test(self):
        return np.array([self.predict(0) for _ in range(len(self.train_y))])

    def calc_loss(self, predictions):
        test_y = [list(rows.values) for index, rows in self.train_y.iterrows()]
        return calc_loss(test_y, predictions)