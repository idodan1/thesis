import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2 as cv
import pandas as pd
from PIL import Image
from tensorflow.keras import activations
import pickle


def calc_loss(y_true, y_pred):
    return tf.keras.losses.MeanAbsolutePercentageError()(y_true, y_pred).numpy()


def calc_per_point(y_true, y_pred):
    return [(sum([abs(y_pred[i][j] - y_true[i][j]) for j in range(len(y_pred[i]))])) / 3 for i in
            range(len(y_pred))]


def select_features(x, feature_list):
    return [[item[j] for j in range(len(feature_list)) if feature_list[j] == 1] for item in x]


def plot_textures(texture_df, loss_by_point=0):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    silt, clay, sand = texture_df['silt'].values, texture_df['clay'].values, texture_df['sand'].values
    axes[0].scatter(silt, clay)
    axes[0].set_xlim(min(silt) - 5, max(silt) + 5)
    axes[0].set_xlabel('silt percent')
    axes[0].set_ylabel('clay percent')
    axes[0].set_ylim(min(clay) - 5, max(clay) + 5)
    lst = [silt, clay, sand]
    names = ['silt', 'clay', 'sand']
    colors = ['r', 'g', 'b']
    lines = []
    for i in range(3):
        line = axes[1].scatter(list(range(1, len(lst[i]) + 1)), lst[i], c=colors[i], label=names[i])
        lines.append(line)
    if loss_by_point != 0:
        ax2 = axes[1].twinx()
        ax2.set_ylabel('loss')
        line = ax2.scatter(list(range(1, len(lst[i]) + 1)), loss_by_point, c='y', label='loss by point')

        lines.append(line)
    axes[1].set_xlabel('point index')
    axes[1].set_ylabel('percent %')
    axes[1].legend(handles=lines, loc='upper right', bbox_to_anchor=(0.95, 0.8))
    fig.tight_layout()
    plt.show()


def open_image(adress):
    img = cv.imread(adress)
    img_convert = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img_convert


def augment_flip(img):
    flipped1 = np.array([[[img[k, j, i] for i in range(len(img[k, j]))]for j in range(len(img[0])-1, -1, -1)] for k in range(len(img))])
    flipped2 = [[[img[k, j, i] for i in range(len(img[k, j]))]for j in range(len(img[0]))] for k in range(len(img)-1, -1, -1)]
    flipped3 = [[[img[k, j, i] for i in range(len(img[k, j]))]for j in range(len(img[0])-1, -1, -1)] for k in range(len(img)-1, -1, -1)]
    return [img, np.array(flipped1), np.array(flipped2), np.array(flipped3)]


def create_data(ad, df_texture):
    label = (ad.split("/")[-1]).split("_")[0]  # cut the label from location
    line = df_texture.loc[df_texture['sample'].astype(str) == label]
    if not line.empty:
        label = [float(line['sand']), float(line['silt']), float(line['clay'])]
        img = open_image(ad)
        img = np.asarray(img, dtype=np.int8)
    return np.array(img).astype(int), np.array(label)


def create_photos_dir():
    edge_val = 512
    imgs_names = glob.glob('images/*.JPG')
    counter = 0
    been_there = [False]*63
    for ad in imgs_names:
        print("did {0} photos".format(counter))
        counter += 1
        only_name = ad.split("/")[-1]
        only_name = only_name.split(".")[0]
        only_name = only_name[:-4]
        if not been_there[int(only_name)-1]:
            been_there[int(only_name)-1] = True
            img = open_image(ad)
            img = np.asarray(img, dtype=np.int8)
            img = img[edge_val:-edge_val, edge_val:-edge_val]
            images = augment_flip(img)
            for i in range(len(images)):
                to_save = Image.fromarray(images[i].astype(np.uint8))
                to_save.save('images_cut_n_augment/{0}_{1}.png'.format(only_name, i))


def create_mini_images(img_arr, mini_image_size):
    mini_images_lst = []
    for i in range(0, len(img_arr), mini_image_size):
        image = img_arr[i:i+mini_image_size, i:i+mini_image_size]
        mini_images_lst.append(image)
    return np.array(mini_images_lst)


def create_df_n_lists_for_model():
    all_data_file = 'soil_data_2020_all data.xlsx'
    all_data_df = pd.read_excel(all_data_file, index_col=0)[2:]

    with open('train_nums_net', 'rb') as f:
        train_nums = pickle.load(f)
    with open('val_nums_net', 'rb') as f:
        val_nums = pickle.load(f)
    with open('test_nums', 'rb') as f:
        test_nums = pickle.load(f)
    with open('cols_for_model', 'rb') as f:
        cols_for_model = pickle.load(f)
    with open('texture_master_cols', 'rb') as f:
        texture_master_cols = pickle.load(f)
    with open('texture_hydro_cols', 'rb') as f:
        texture_hydro_cols = pickle.load(f)

    activation_funcs = [activations.sigmoid, activations.relu, activations.linear]
    return all_data_df, train_nums, val_nums, test_nums, cols_for_model, texture_master_cols, \
           texture_hydro_cols, activation_funcs


def write_results(model, cols_for_model, best_loss, best_member, best_member_net, best_activation,
                  texture_name, activation_funcs):
    results_df = pd.read_excel('results net.xlsx', index=False)
    if str(model) not in results_df['model name'].values or \
            (results_df.loc[results_df['model name'] == str(model)])['texture type'].values[0] != texture_name:
        model_cols = [cols_for_model[i] for i in range(len(cols_for_model)) if best_member[i] == 1]
        row_df = pd.DataFrame([[str(model), best_loss, texture_name, str(model_cols), str(best_member_net),
                                str(best_activation)]], columns=results_df.columns)
        results_df = pd.concat([results_df, row_df], ignore_index=True)
    else:
        if (results_df.loc[(results_df['model name'] == str(model)) &
                           (results_df['texture type'] == texture_name)])['model loss'].values[0] > best_loss:
            index = list(results_df.loc[results_df['model name'] == str(model)].index)[0]
            model_cols = [cols_for_model[i] for i in range(len(cols_for_model)) if best_member[i] == 1]
            results_df.loc[index, 'model loss'] = best_loss
            results_df.loc[index, 'features'] = str(model_cols)
            results_df.loc[index, 'net architecture'] = str(best_member_net)
            results_df.loc[index, 'activations'] = str([activation_funcs[k] for k in best_activation])

    results_df.to_excel('results net.xlsx', index=False)


