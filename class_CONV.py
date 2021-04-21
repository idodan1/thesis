from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dropout, Flatten, Input, Dense, concatenate
from tensorflow.keras import Model
from functions import *
from tensorflow.keras.callbacks import EarlyStopping
import copy


class Conv:
    model_name = 'Conv'

    def __init__(self, train_x, val_x, test_x, train_y, val_y, test_y, feature_dict):
        self.train_x, self.val_x, self.test_x, self.train_y, self.val_y, self.test_y = train_x, val_x, test_x, train_y, \
                                                                                       val_y, test_y
        self.feature_dict = feature_dict
        image_sizes = [20, 50, 100]
        self.image_size = image_sizes[self.feature_dict['mini_image_size_index']]
        img_size = 2000
        self.num_of_mini_img = img_size // self.image_size
        with open('train nums', 'rb') as f:
            self.train_nums = pickle.load(f)

    def create_model(self):
        # create net for images
        activation_funcs = [activations.sigmoid, activations.relu, activations.linear]
        num_neurons = 10 * self.feature_dict['num_neurons']

        def add_conv_block(input_layer, num_filters, problem=False):
            _x = Conv2D(num_filters, 3, activation='relu', padding='same')(input_layer)
            _x = BatchNormalization()(_x)
            if problem:
                _x = Conv2D(num_filters, 3, activation='relu', padding='same')(_x)
            else:
                _x = Conv2D(num_filters, 3, activation='relu')(_x)
            _x = MaxPooling2D(pool_size=2)(_x)
            _x = Dropout(0.5)(_x)
            return _x
        input_img = Input(shape=(self.image_size, self.image_size, 3))
        num_of_filters = 2 ** self.feature_dict['num_of_filters_power']
        x = add_conv_block(input_img, num_of_filters)
        for i in range(1, self.feature_dict['num_of_blocks']):
            num_of_filters *= 2
            try:
                x = add_conv_block(x, num_of_filters)
            except:
                x = add_conv_block(x, num_of_filters, problem=True)
        x = Flatten()(x)
        x = Dense(num_neurons)(x)
        x = Model(inputs=input_img, outputs=x)

        # create net for numeric data
        neurons_constant = len(self.train_x.columns)
        input_num = Input(shape=neurons_constant)
        num_of_neurons = self.feature_dict["num_of_neurons"]
        activation_nums = self.feature_dict["activations"]
        y = Dense(num_of_neurons[0] * neurons_constant, activation=activation_funcs[activation_nums[0]],
                  kernel_initializer='random_uniform', bias_initializer='random_uniform')(input_num)
        for i in range(1, len(num_of_neurons)):
            y = Dense(num_of_neurons[i]*neurons_constant, activation=activation_funcs[activation_nums[i]],
                  kernel_initializer='random_uniform', bias_initializer='random_uniform')(input_num)
        y = Model(inputs=input_num, outputs=y)

        combined = concatenate([x.output, y.output])
        z = Dense(30, activation='relu',
                  kernel_initializer='random_uniform', bias_initializer='random_uniform')(combined)
        z = Dense(3, activation='linear',
                  kernel_initializer='random_uniform', bias_initializer='random_uniform')(z)

        self.model = Model(inputs=[input_img, input_num], outputs=z)
        self.model.compile(
            # loss=tf.keras.losses.MeanAbsolutePercentageError(),
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer='adam', metrics=['accuracy']
        )

    def train(self):
        monitor_list = ['loss', 'val_loss']
        img_location = 'images_cut_n_augment'
        images_names = np.array(glob.glob(img_location + '/*.png'))
        val_set_size = self.feature_dict['val_set_size']
        self.train_names, self.val_names, self.test_names = divide_images(images_names, val_set_size, self.train_nums)
        val_data = names_to_arr_num(self.val_names, self.train_x)
        val_x_img, val_x_num, val_y = names_to_arr_img(self.val_names, self.train_y, val_data, self.image_size,
                                                       self.num_of_mini_img)

        stop_when_enough = EarlyStopping(monitor='loss', min_delta=0, patience=5,
                                         restore_best_weights=True)
        lowest_loss = np.inf
        best_weights = 0
        counter_early_stopping = 0
        total_counter = 0
        while total_counter < 50:
            total_counter += 1
            s = np.random.choice(range(len(self.train_names)), self.feature_dict['batch_size'])
            numeric_data = names_to_arr_num(self.train_names[s], self.train_x)
            batch_x_img, batch_x_num, batch_y = names_to_arr_img(self.train_names[s], self.train_y, numeric_data,
                                                                 self.image_size, self.num_of_mini_img)
            p = np.random.permutation(len(batch_y))
            batch_x_img = batch_x_img[p]
            batch_y = batch_y[p]
            self.history = self.model.fit(
                x=[batch_x_img, batch_x_num], y=batch_y,
                validation_data=([val_x_img, val_x_num], val_y),
                epochs=self.feature_dict['epochs'], batch_size=self.feature_dict['batch_size'], verbose=0,
                callbacks=stop_when_enough
            )
            training_by = monitor_list[self.feature_dict['training_by']]
            min_loss = 30
            if lowest_loss > self.history.history[training_by][-1] and min_loss > self.history.history[training_by][-1]:
                lowest_loss = self.history.history[training_by][-1]
                best_weights = copy.deepcopy(self.model.get_weights())
                counter_early_stopping = 0
            elif counter_early_stopping == self.feature_dict['early_stopping'] or lowest_loss > min_loss:
                break
            else:
                counter_early_stopping += 1
        if type(best_weights) != int:
            self.model.set_weights(best_weights)

    def predict(self, x):
        predictions = self.model.predict([x])
        gap = [(np.max(predictions[:, i]) - np.min(predictions[:, i])) for i in range(3)]
        prediction = [np.mean(predictions[:, 0]), np.mean(predictions[:, 1]), np.mean(predictions[:, 2])]
        total = sum(prediction) / 100
        norm = [p / total for p in prediction]
        return norm, gap

    def predict_test(self):
        all_predictions = []
        gaps_test = []
        all_y = []
        test_names = forget_augmentation(self.test_names)
        for i in range(len(test_names)):
            x, y = create_data(test_names[i], self.test_y)
            x_mini = create_mini_images(x, self.image_size)
            numeric_data = np.array(names_to_arr_num([test_names[i]], self.test_x)[0])
            numeric_data = np.array([numeric_data for _ in range(x_mini.shape[0])])
            predictions, gap = self.predict([x_mini, numeric_data])
            all_predictions.append(predictions)
            gaps_test.append(gap)
            all_y.append(y)
        return all_predictions

    def predict_train(self):
        all_predictions = []
        gaps_test = []
        all_y = []
        train_names = list(self.train_names) + list(self.val_names)
        train_names = forget_augmentation(train_names)
        for i in range(len(train_names)):
            x, y = create_data(train_names[i], self.train_y)
            x_mini = create_mini_images(x, self.image_size)
            numeric_data = np.array(names_to_arr_num([train_names[i]], self.train_x)[0])
            numeric_data = np.array([numeric_data for _ in range(x_mini.shape[0])])
            predictions, gap = self.predict([x_mini, numeric_data])
            all_predictions.append(predictions)
            gaps_test.append(gap)
            all_y.append(y)
        return all_predictions

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)

    def calc_loss(self, predictions, test_y):
        # test_y = [list(rows.values) for index, rows in test_y.iterrows()]
        return calc_loss(test_y, predictions)


def divide_images(images_names, val_set_size, train_num):
    val_num = np.random.choice(list(train_num), val_set_size)
    train_num = list(set(train_num) - set(val_num))
    train_names = []
    val_names = []
    test_names = []
    for img in images_names:
        img_num = int((img.split("/")[1]).split("_")[0])
        if img_num in train_num:
            train_names.append(img)
        elif img_num in val_num:
            val_names.append(img)
        else:
            test_names.append(img)
    return np.array(train_names), np.array(val_names), np.array(test_names)


def names_to_arr_img(sample_names, df_texture, numeric_data, mini_img_size, num_of_mini_images):
    x_mini, x_num_data, y_res = [], [], []
    for i in range(len(sample_names)):
        x, y = create_data(sample_names[i], df_texture)
        x_mini.extend(create_mini_images(x, mini_img_size))
        x_num_data.extend([numeric_data[i] for _ in range(num_of_mini_images)])
        y_res.extend([y for _ in range(num_of_mini_images)])
    return np.array(x_mini), np.array(x_num_data), np.array(y_res)


def names_to_arr_img_only_img(sample_names, df_texture, mini_img_size, num_of_mini_images):
    x_mini, y_res = [], []
    for i in range(len(sample_names)):
        x, y = create_data(sample_names[i], df_texture)
        x_mini.extend(create_mini_images(x, mini_img_size))
        y_res.extend([y for _ in range(num_of_mini_images)])
    return np.array(x_mini), np.array(y_res)


def names_to_arr_num(sample_names, df_num):
    res = []
    for ad in sample_names:
        label = (ad.split("/")[-1]).split("_")[0]  # cut the label from location
        line = df_num.ix[int(label)]
        res.append(list(line.values))
    return np.array(res)


def main():
    train_x_num = pd.read_excel("train_x.xlsx", index_col=0).sort_values(by=['sample'])
    test_x_num = pd.read_excel("test_x.xlsx", index_col=0).sort_values(by=['sample'])

    location = 'images_cut_n_augment'
    models_path = 'models/train_model'
    df_texture = pd.read_excel('texture df.xlsx')
    img_size = 2000
    mini_img_size = 100
    num_of_mini_img = img_size//mini_img_size
    val_set_size = 5
    numeric_input_len = 14
    images_names = np.array(glob.glob(location + '/*.png'))
    train_names, val_names, test_names = divide_images(images_names, val_set_size)

    model = Conv()
    model.create_model(mini_img_size, numeric_input_len)
    batch_size = 5
    num_iter = 3
    epochs = 3

    val_data = names_to_arr_num(val_names, train_x_num)
    val_x_img, val_x_num, val_y = names_to_arr_img(val_names, df_texture, val_data, mini_img_size, num_of_mini_img)

    history = []
    for i in range(num_iter):
        s = np.random.choice(range(len(train_names)), batch_size)
        numeric_data = names_to_arr_num(train_names[s], train_x_num)
        batch_x_img, batch_x_num, batch_y = names_to_arr_img(train_names[s], df_texture, numeric_data, mini_img_size, num_of_mini_img)
        p = np.random.permutation(len(batch_y))
        batch_x_img, batch_x_num, batch_y = batch_x_img[p], batch_x_num[p], batch_y[p]

        model.train(batch_x_img, batch_x_num, batch_y, val_x_img, val_x_num, val_y, batch_size, epochs)

        if i % 5 == 0:
            print("num of iter = {0}".format(i))
            print("train acc = {0}, val acc = {1}".format(model.history.history['accuracy'][-1], model.history.history['val_accuracy'][-1]))
            print("train loss = {0}, val loss = {1}".format(model.history.history['loss'][-1], model.history.history['val_loss'][-1]))
            print()
        history.append([model.history.history['loss'][-1], model.history.history['val_loss'][-1]])

    # model.save_model(models_path)

    # plot loss over iterations
    fig, axes = plt.subplots(2, 1)
    iter_range = list(range(num_iter))
    train = [item[0] for item in history]
    val = [item[1] for item in history]
    axes[0].plot(iter_range, train)
    axes[1].plot(iter_range, val)
    axes[0].set_title('train loss')
    axes[1].set_title('validation loss')
    axes[0].set_ylabel('loss')
    axes[1].set_ylabel('loss')
    axes[0].set_xlabel('iter')
    axes[1].set_xlabel('iter')
    # plt.show()

    # test data
    loss = 0
    all_predictions = []
    all_y = []
    print(len(test_names))
    for i in range(len(test_names)):
        x, y = create_data(test_names[i], df_texture)
        x_mini = create_mini_images(x, mini_img_size)
        numeric_data = np.array(names_to_arr_num([test_names[i]], test_x_num)[0])
        numeric_data = np.array([numeric_data for _ in range(x_mini.shape[0])])
        predictions = model.predict([x_mini, numeric_data])
        all_predictions.append(predictions)
        all_y.append(y)
        # p_loss = calc_loss(predictions, y)
        # loss += p_loss
    loss = calc_loss(all_y, all_predictions)
    print(loss)

# main()



"""
i was in the middle of doing all over again training, pay attention to both numeric and image.
notice that every thing is the right data type. for parameters from the winners in con img and numeric
we need to write functions that gets the parameters  
"""