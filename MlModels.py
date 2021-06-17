from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from functions import *
from tensorflow.keras.layers import Dropout, Flatten, Input, Dense, Conv2D, MaxPooling2D, BatchNormalization,\
    concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import activations, Model
import copy


class MlModel:
    def __init__(self, train_x, test_x, train_y, test_y, features_dict=None):
        self.train_x, self.test_x, self.train_y, self.test_y = train_x, test_x, train_y, test_y
        self.model = None
        self.features_dict = features_dict

    def train(self):
        self.model.fit(self.train_x, self.train_y)

    def predict(self, df_to_predict):
        return pd.DataFrame(data=self.model.predict(df_to_predict), columns=self.train_y.columns,
                            index=df_to_predict.index)

    @staticmethod
    def normalize(predictions):
        cols = predictions.columns
        predictions['sum'] = predictions.sum(axis=1)
        # normalize to 100
        for c in cols:
            predictions[c] = predictions[c] / predictions['sum'] * 100
        return predictions

    def predict_test(self):
        return self.normalize(self.predict(self.test_x))

    def predict_train(self):
        return self.normalize(self.predict(self.train_x))

    def save(self, path):
        pickle.dump(self.model, open(path + '.sav', 'wb'))

    def load(self, path):
        self.model = pickle.load(open(path + '.sav', 'rb'))


class LinearReg(MlModel):
    model_name = 'Linear Regression'

    def create_model(self):
        self.model = LinearRegression()


class RandomForest(MlModel):
    model_name = 'Random Forest'

    def create_model(self, max_depth=10):
        self.model = RandomForestRegressor(max_depth=max_depth, random_state=0)


class NN(MlModel):
    model_name = 'NN'
    activation_funcs = [activations.sigmoid, activations.relu, activations.linear]
    feature_space = {"batch_size": [3, 10], "num_of_neurons": [1, 10, list], 'activations': [0, 3, list],
                     "restore_best_weights": [0, 1], "patience": [10, 50], 'num_of_layers': [1, 7]}

    def __init__(self, train_x, test_x, train_y, test_y, feature_dict):
        super(NN, self).__init__(train_x, test_x, train_y, test_y)
        self.feature_dict = feature_dict

    def create_model(self):
        self.model = tf.keras.models.Sequential()
        self.model.add(Input(shape=len(self.train_x.columns)))
        neurons_constant = len(self.train_x.columns)  # num of neurons is proportional to the num of features
        num_of_neurons = self.feature_dict["num_of_neurons"]
        activation_nums = self.feature_dict["activations"]
        for i in range(len(num_of_neurons)):
            self.model.add(Dense(num_of_neurons[i] * neurons_constant,
                                 activation=self.activation_funcs[activation_nums[i]]))
        self.model.add(Dense(self.train_y.shape[1]))

        self.model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer='adam', metrics=['accuracy']
        )

    def train(self):
        x = self.train_x.values.astype(np.float)
        y = self.train_y.values.astype(np.float)
        stop_when_enough = EarlyStopping(monitor='loss', min_delta=0,
                                         patience=self.feature_dict['patience'],
                                         restore_best_weights=(self.feature_dict['restore_best_weights'] == 1))
        self.model.fit(
            x, y,
            validation_data=(x, y),
            epochs=2000, batch_size=self.feature_dict['batch_size'], verbose=0, callbacks=stop_when_enough
        )

    def predict(self, df_to_predict):
        return pd.DataFrame(data=self.model.predict(df_to_predict.astype(np.float)), columns=self.train_y.columns,
                            index=df_to_predict.index)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)


class ConvImg(NN):
    model_name = 'conv_img'
    img_size = 2000
    img_location = 'images_cut_n_augment'
    feature_space = {"batch_size": [3, 10], 'early_stopping': [1, 10], 'num_of_blocks': [1, 6],
                     'num_of_filters_power': [1, 4], 'epochs': [1, 7], 'num_neurons': [1, 10], 'mini_image_size_index':
                         [0, 3], 'activation': [0, 3]}

    def __init__(self, train_x, test_x, train_y, test_y, feature_dict):
        train_img, test_img = self.divide_images(train_x.index, test_x.index)
        super(ConvImg, self).__init__(train_img, test_img, train_y, test_y, feature_dict)
        image_sizes = [20, 50, 100]
        self.image_size = image_sizes[self.feature_dict['mini_image_size_index']]
        self.num_of_mini_img = self.img_size // self.image_size

    @staticmethod
    def get_label_from_name(name):
        return (name.split("/")[-1]).split("_")[0]  # cut the label from path

    @staticmethod
    def add_conv_block(input_layer, num_filters):
        x = Conv2D(num_filters, 3, activation='relu', padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = Conv2D(num_filters, 3, activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.5)(x)
        return x

    def create_conv_model(self):
        num_neurons = 10*self.feature_dict['num_neurons']
        input_img = Input(shape=(self.image_size, self.image_size, 3))
        num_of_filters = 2 ** self.feature_dict['num_of_filters_power']
        x = self.add_conv_block(input_img, num_of_filters)
        for i in range(1, self.feature_dict['num_of_blocks']):
            num_of_filters *= 2
            try:
                x = self.add_conv_block(x, num_of_filters)
            except ValueError:  # some combinations do not have legal architecture so we must break
                break
        x = Flatten()(x)
        x = Dense(num_neurons)(x)
        x = Model(inputs=input_img, outputs=x)
        return input_img, x

    def create_model(self):
        num_neurons = 10*self.feature_dict['num_neurons']
        input_img, x = self.create_conv_model()
        # The kernel, the bias and the linear (in the second) is because it goes to zero for some reason
        z = Dense(num_neurons, activation=self.activation_funcs[self.feature_dict['activation']],
                  kernel_initializer='random_uniform', bias_initializer='random_uniform')(x.output)
        z = Dense(self.train_y.shape[1], activation='linear',
                  kernel_initializer='random_uniform', bias_initializer='random_uniform')(z)
        self.model = Model(inputs=[input_img], outputs=z)

        self.model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer='adam', metrics=['accuracy']
        )

    def divide_images(self, train_num, test_num):
        images_names = np.array(glob.glob(self.img_location + '/*.png'))
        train_names = [img for img in images_names if int(self.get_label_from_name(img)) in train_num]
        test_names = [img for img in images_names if int(self.get_label_from_name(img)) in test_num]
        return np.array(train_names), np.array(test_names)

    def get_image(self, image_path):
        img = cv.imread(image_path)
        img_convert = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        return self.create_mini_images(np.asarray(img_convert, dtype=np.int8))

    def create_mini_images(self, img_arr):
        return np.array([img_arr[i:i + self.image_size, i:i + self.image_size] for i in
                         range(0, len(img_arr), self.image_size)])

    def create_y(self, image_path):
        label = self.get_label_from_name(image_path)  # cut the label from path
        line = self.train_y.ix[int(label)]
        return pd.DataFrame(columns=self.train_y.columns).append([line]*self.num_of_mini_img, ignore_index=True)

    def create_batch(self):
        sample = np.random.choice(self.train_x, self.feature_dict['batch_size'])
        x = np.array([item for s in sample for item in self.get_image(s)]).astype(np.float32)
        y = [self.create_y(s) for s in sample]
        y = np.array([item for df in y for item in df.values]).astype(np.float32)
        p = np.random.permutation(len(y))
        return x[p], y[p]

    def train(self):
        stop_when_enough = EarlyStopping(monitor='loss', min_delta=0, patience=5,
                                         restore_best_weights=True)
        lowest_loss = np.inf
        best_weights = None
        counter_early_stopping = 0
        highest_loss_tolerable = 30
        max_number_of_iterations = 50  # check this, maybe too low
        num_iter = 0
        for _ in range(max_number_of_iterations):
            num_iter += 1
            batch_x, batch_y = self.create_batch()
            history = self.model.fit(
                x=batch_x, y=batch_y,
                validation_data=(batch_x, batch_y),
                epochs=self.feature_dict['epochs'], batch_size=self.feature_dict['batch_size'], verbose=0,
                callbacks=stop_when_enough
                )
            if lowest_loss > history.history['loss'][-1] and highest_loss_tolerable > history.history['loss'][-1]:
                lowest_loss = history.history['loss'][-1]
                best_weights = copy.deepcopy(self.model.get_weights())
                counter_early_stopping = 0
            elif counter_early_stopping == self.feature_dict['early_stopping'] or lowest_loss > highest_loss_tolerable:
                break
            else:
                counter_early_stopping += 1

        print('num iterations in training conv new:', num_iter)
        if best_weights is not None:
            self.model.set_weights(best_weights)

    def create_data_to_predict(self, to_predict):
        names_to_predict = forget_augmentation(to_predict)
        labels = [int(self.get_label_from_name(image_path)) for image_path in names_to_predict]
        return [self.get_image(name) for name in names_to_predict], labels

    def predict(self, to_predict):
        data_to_predict, labels = self.create_data_to_predict(to_predict)
        all_predictions = []
        for i in range(len(data_to_predict)):
            predictions = self.model.predict(data_to_predict[i])
            prediction = [np.mean(predictions[:, 0]), np.mean(predictions[:, 1]), np.mean(predictions[:, 2])]
            all_predictions.append(prediction)
        return pd.DataFrame(data=all_predictions, columns=self.train_y.columns,
                            index=labels)


class Conv(ConvImg):
    model_name = 'Conv'
    img_size = 2000
    feature_space = {"batch_size": [3, 10], 'early_stopping': [1, 10], 'num_of_blocks': [1, 6],
                     'num_of_filters_power': [1, 4], 'epochs': [1, 7], 'num_neurons': [1, 10], 'mini_image_size_index':
                         [0, 3], 'activation': [0, 3], 'num_of_neurons': [1, 10, list], 'activations': [0, 3, list],
                     'num_of_layers': [1, 7]}

    def __init__(self, train_x, test_x, train_y, test_y, feature_dict):
        super(Conv, self).__init__(train_x, test_x, train_y, test_y, feature_dict)
        self.numeric_df = pd.concat([train_x, test_x], sort=False)
        image_sizes = [20, 50, 100]
        self.image_size = image_sizes[self.feature_dict['mini_image_size_index']]
        self.num_of_mini_img = self.img_size // self.image_size

    def create_numeric_model(self):
        activation_funcs = [activations.sigmoid, activations.relu, activations.linear]
        neurons_constant = len(self.numeric_df.columns)
        input_num = Input(shape=neurons_constant)
        num_of_neurons = self.feature_dict["num_of_neurons"]
        activation_nums = self.feature_dict["activations"]
        y = Dense(num_of_neurons[0] * neurons_constant, activation=activation_funcs[activation_nums[0]],
                  kernel_initializer='random_uniform', bias_initializer='random_uniform')(input_num)
        for i in range(1, len(num_of_neurons)):
            y = Dense(num_of_neurons[i] * neurons_constant, activation=activation_funcs[activation_nums[i]],
                      kernel_initializer='random_uniform', bias_initializer='random_uniform')(input_num)
        return input_num, Model(inputs=input_num, outputs=y)

    def create_model(self):
        input_img, x = self.create_conv_model()
        input_num, y = self.create_numeric_model()

        # The kernel, the bias and the linear (in the second) is because it goes to zero for some reason
        combined = concatenate([x.output, y.output])
        z = Dense(30, activation='relu',
                  kernel_initializer='random_uniform', bias_initializer='random_uniform')(combined)
        z = Dense(3, activation='linear',
                  kernel_initializer='random_uniform', bias_initializer='random_uniform')(z)

        self.model = Model(inputs=[input_img, input_num], outputs=z)
        self.model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer='adam', metrics=['accuracy']
        )

    def create_x_num_single(self, image_path):
        label = self.get_label_from_name(image_path)  # cut the label from path
        line = self.numeric_df.ix[int(label)]
        return pd.DataFrame(columns=self.numeric_df.columns).append([line] * self.num_of_mini_img, ignore_index=True)

    def create_x_num(self, sample):
        x_num = [self.create_x_num_single(s) for s in sample]
        return np.array([item for df in x_num for item in df.values]).astype(np.float)

    def create_batch(self):
        sample = np.random.choice(self.train_x, self.feature_dict['batch_size'])
        x_img = np.array([item for s in sample for item in self.get_image(s)]).astype(np.float32)
        x_num = self.create_x_num(sample)
        y = [self.create_y(s) for s in sample]
        y = np.array([item for df in y for item in df.values]).astype(np.float32)
        p = np.random.permutation(len(y))
        return [x_img[p], x_num[p]], y[p]

    def create_data_to_predict(self, to_predict):
        names_to_predict = forget_augmentation(to_predict)
        labels = [int(self.get_label_from_name(image_path)) for image_path in names_to_predict]
        image_data = [self.get_image(name) for name in names_to_predict]
        numeric_data = self.create_x_num(names_to_predict)
        data_together = [[image_data[i], numeric_data[i*self.num_of_mini_img: (i+1)*self.num_of_mini_img]] for i in
                         range(len(image_data))]
        return data_together, labels







