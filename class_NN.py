from tensorflow.keras.layers import Input, Dense
from functions import *
from tensorflow.keras.callbacks import EarlyStopping


class NN:
    model_name = 'NN'

    def __init__(self, train_x, val_x, test_x, train_y, val_y, test_y, feature_dict):
        self.train_x, self.val_x, self.test_x, self.train_y, self.val_y, self.test_y = train_x, val_x, test_x, train_y,\
                                                                                       val_y, test_y
        self.feature_dict = feature_dict

    def create_model(self):
        activation_funcs = [activations.sigmoid, activations.relu, activations.linear]
        self.model = tf.keras.models.Sequential()
        self.model.add(Input(shape=len(self.train_x.columns)))
        neurons_constant = len(self.train_x.columns)
        num_of_neurons = self.feature_dict["num_of_neurons"]
        activation_nums = self.feature_dict["activations"]
        for i in range(len(num_of_neurons)):
            self.model.add(Dense(num_of_neurons[i] * neurons_constant, activation=activation_funcs[activation_nums[i]]))
        self.model.add(Dense(3))

        self.model.compile(
            # loss=tf.keras.losses.MeanAbsolutePercentageError(),
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer='adam', metrics=['accuracy']
        )

    def train(self):
        monitor_list = ['loss', 'val_loss']
        x = [list(rows.values) for index, rows in self.train_x.iterrows()]
        y = [list(rows.values) for index, rows in self.train_y.iterrows()]
        val_x = [list(rows.values) for index, rows in self.val_x.iterrows()]
        val_y = [list(rows.values) for index, rows in self.val_y.iterrows()]
        stop_when_enough = EarlyStopping(monitor=monitor_list[self.feature_dict['monitor']], min_delta=0,
                                         patience=self.feature_dict['patience'],
                                         restore_best_weights=(self.feature_dict['restore_best_weights'] == 1))
        self.history = self.model.fit(
            x, y,
            validation_data=(val_x, val_y),
            epochs=2000, batch_size=self.feature_dict['batch_size'], verbose=0, callbacks=stop_when_enough
        )

    def predict(self, x):
        try:
            predictions = self.model.predict([x])[0]
        except:
            predictions = self.model.predict([list(x)])[0]
        total = sum(predictions) / 100
        return [p / total for p in predictions]

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
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)




