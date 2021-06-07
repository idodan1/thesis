from functions import *


class RandomModel:
    def __init__(self, train_y, test_y):
        self.train_y, self.test_y = train_y, test_y

    def create_model(self):
        self.means, self.std = [self.train_y.mean()], [self.train_y.std()]

    def predict(self):
        predictions = [np.random.normal(self.means[i], self.std[i]) for i in range(len(self.means))][0]
        total = sum(predictions)/100
        return [p/total for p in predictions]

    def predict_test(self):
        return np.array([self.predict() for _ in range(len(self.test_y))])

    def calc_loss(self, predictions):
        test_y = [list(rows.values) for index, rows in self.test_y.iterrows()]
        return calc_loss(test_y, predictions)