from sklearn.ensemble import RandomForestRegressor
from functions import *


class RandomForest:
  model_name = 'Random Forest'

  def __init__(self, train_x, test_x, train_y, test_y):
    self.train_x, self.test_x, self.train_y, self.test_y = train_x, test_x, train_y, test_y

  def create_model(self, max_depth=10):
    self.model = RandomForestRegressor(max_depth=max_depth, random_state=0)

  def train(self):
    self.model.fit(self.train_x, self.train_y)

  def predict_test(self):
    predictions = pd.DataFrame(data=self.model.predict(self.test_x), columns=self.train_y.columns, index=self.test_x.index)
    cols = predictions.columns
    predictions['sum'] = predictions.sum(axis=1)
    # normalize to 100
    for c in cols:
      predictions[c] = predictions[c]/predictions['sum']*100
    return predictions

  def predict_train(self):
    predictions = pd.DataFrame(data=self.model.predict(self.train_x), columns=self.train_y.columns,
                               index=self.train_x.index)
    cols = predictions.columns
    predictions['sum'] = predictions.sum(axis=1)
    # normalize to 100
    for c in cols:
      predictions[c] = predictions[c] / predictions['sum'] * 100
    return predictions

  def save(self, path):
    pickle.dump(self.model, open(path + '.sav', 'wb'))

  def load(self, path):
    self.model = pickle.load(open(path + '.sav', 'rb'))

