from sklearn.linear_model import LinearRegression
from functions import *


class LinearReg:
  model_name = 'Linear Regression'

  def __init__(self, train_x, test_x, train_y, test_y):
    self.train_x, self.test_x, self.train_y, self.test_y = train_x, test_x, train_y, test_y

  def create_model(self):
    self.model = LinearRegression()

  def train(self):
    self.model.fit(self.train_x, self.train_y)

  def predict(self, x):
    predictions = self.model.predict([x])[0]
    total = sum(predictions)/100
    return [p/total for p in predictions]

  def predict_test(self):
    cols = [c + '_prediction' for c in self.test_y.columns]
    predictions = pd.DataFrame(data=self.model.predict(self.test_x), columns=cols, index=self.test_x.index)
    cols = predictions.columns
    predictions['sum'] = predictions.sum(axis=1)
    # normalize to 100
    for c in cols:
      predictions[c] = predictions[c]/predictions['sum']*100
    self.test_y[cols] = predictions[cols]

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