from sklearn.ensemble import RandomForestRegressor
from functions import *


class RandomForest:
  model_name = 'Random Forest'
  def __init__(self, train_x, test_x, train_y, test_y):
    self.train_x, self.test_x, self.train_y, self.test_y = train_x, test_x, train_y, test_y

  def create_model(self, max_depth=10):
    self.model = RandomForestRegressor(max_depth=max_depth, random_state=0)

  def train(self):
    x = [list(rows.values) for index, rows in self.train_x.iterrows()]
    y = [list(rows.values) for index, rows in self.train_y.iterrows()]
    self.model.fit(x, y)

  def predict(self, x):
    predictions = self.model.predict([x])[0]
    total = sum(predictions)/100
    return [p/total for p in predictions]

  def predict_test(self):
    test_x = [list(rows.values) for index, rows in self.test_x.iterrows()]
    return [self.predict(x) for x in test_x]

  def predict_train(self):
    train_x = [list(rows.values) for index, rows in self.train_x.iterrows()]
    return [self.predict(x) for x in train_x]

  def calc_loss(self, predictions):
    test_y = [list(rows.values) for index, rows in self.test_y.iterrows()]
    return calc_loss(test_y, predictions)

  def save(self, path):
    pickle.dump(self.model, open(path + '.sav', 'wb'))

  def load(self, path):
    self.model = pickle.load(open(path + '.sav', 'rb'))

