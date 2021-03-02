import numpy as np


class RandomModel:
    def __init__(self, train_x, test_x, train_y, test_y, features_list):
        self.train_y, self.test_y = train_y, test_y

    def create_model(self):
        self.cols = [[self.train_y[i][j] for i in range(len(self.train_y))] for j in range(len(self.train_y[0]))]
        self.means, self.std = [np.mean(self.cols[i]) for i in range(len(self.cols))], [np.std(self.cols[i]) for i in
                                                                                        range(len(self.cols))]

    def train(self):
        return

    def predict(self, x):
        predictions = [np.random.normal(self.means[i], self.std[i]) for i in range(len(self.means))]
        total = sum(predictions)/100
        return [p/total for p in predictions]

    def predict_test(self):
        return [self.predict(0) for _ in range(len(self.train_y))]