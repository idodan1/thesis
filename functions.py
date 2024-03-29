import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
from PIL import Image
from tensorflow.keras import activations
import pickle
import os
import ast
import cv2 as cv


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


def augment_flip(img):
    flipped1 = np.array([[[img[k, j, i] for i in range(len(img[k, j]))]for j in range(len(img[0])-1, -1, -1)] for k in range(len(img))])
    flipped2 = [[[img[k, j, i] for i in range(len(img[k, j]))]for j in range(len(img[0]))] for k in range(len(img)-1, -1, -1)]
    flipped3 = [[[img[k, j, i] for i in range(len(img[k, j]))]for j in range(len(img[0])-1, -1, -1)] for k in range(len(img)-1, -1, -1)]
    return [img, np.array(flipped1), np.array(flipped2), np.array(flipped3)]


def create_data(ad, df_texture):
    label = (ad.split("/")[-1]).split("_")[0]  # cut the label from location
    line = df_texture.ix[int(label)]
    label = [float(line[col]) for col in df_texture.columns]
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


def create_df_n_lists_for_model(all_data_df, texture_name):
    with open('train nums {0}'.format(texture_name), 'rb') as f:
        train_nums = pickle.load(f)
    with open('test nums {0}'.format(texture_name), 'rb') as f:
        test_nums = pickle.load(f)
    with open('texture cols master sizer', 'rb') as f:
        texture_master_cols = pickle.load(f)
    with open('texture cols hydro meter', 'rb') as f:
        texture_hydro_cols = pickle.load(f)

    activation_funcs = [activations.sigmoid, activations.relu, activations.linear]
    return train_nums, test_nums, texture_master_cols, texture_hydro_cols, activation_funcs


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


def create_heat_map(df, cols_for_map):
    df = df[cols_for_map]
    df = df.astype(float)
    corr = df.corr()
    res = corr.style.background_gradient(cmap='coolwarm')
    res.to_excel('res.xlsx')


def get_train_test_cols(texture_name):
    texture_cols = get_texture_cols(texture_name)
    train_df, test_df = get_train_test_df()
    with open('winner_cols_{0}'.format(texture_name), 'rb') as f:
        texture_model_cols = pickle.load(f)
    return train_df, test_df, texture_cols, texture_model_cols


def get_train_test_df():
    with open('train nums', 'rb') as f:
        train_nums = pickle.load(f)
    with open('test nums', 'rb') as f:
        test_nums = pickle.load(f)
    all_data_df = get_all_data_df()
    return all_data_df.ix[train_nums], all_data_df.ix[test_nums]


def get_texture_cols(texture_name):
    with open('texture cols {0}'.format(texture_name), 'rb') as f:
        return pickle.load(f)


def get_all_data_df():
    all_data_file = 'soil_data_2020_all data.xlsx'
    return pd.read_excel(all_data_file, index_col=0)[2:]


def calculate_rmse(y_pred, y_real):
    return (sum([(y_pred[i] - y_real[i])**2 for i in range(len(y_pred))])/len(y_pred))**0.5


def calculate_r_square(predictions, observed):
    res = predictions.sub(observed).pow(2).sum()
    tot = observed.sub(observed.mean()).pow(2).sum()
    r2 = 1 - res / tot
    return int(r2*100)/100


def calc_rmses(predictions, measured_df, texture_cols):
    res = pd.DataFrame()
    predictions, measured_df = predictions.sort_index(), measured_df.sort_index()
    for c in texture_cols:
        res[c+'_loss'] = ((measured_df[c] - predictions[c]) ** 2)
    res = (res.mean())**.5

    return res


def forget_augmentation(test_names):
    res = []
    for name in test_names:
        if name.split('.')[0][-1] == '0':
            res.append(name)
    return res


def to_soil_class(threesome):
    """
    code that return the class of a soil sample according to usa triangle from:
     https://www.nrcs.usda.gov/wps/portal/nrcs/detail/soils/survey/?cid=nrcs142p2_054167
    """
    if threesome['clay'] >= 40:
        if 40 <= threesome['silt'] <= 60:
            return 'silty clay'
        elif 45 <= threesome['sand'] <= 60:
            return 'sandy clay'
        else:
            return 'clay'
    elif 35 <= threesome['clay'] <= 40:
        if 45 <= threesome['sand'] <= 65:
            return 'sandy clay'
        elif 20 <= threesome['sand'] <= 45:
            return 'clay loam'
        else:
            return 'silty clay loam'
    elif 27.5 <= threesome['clay'] <= 35:
        if 45 <= threesome['sand'] <= 72.5:
            return 'sandy clay loam'
        elif 20 <= threesome['sand'] <= 45:
            return 'clay loam'
        else:
            return 'silty clay loam'
    elif 20 <= threesome['clay'] <= 27.5:
        if 52.5 <= threesome['sand'] <= 85:
            return 'sandy clay loam'
        elif 22.5 <= threesome['sand'] <= 52.5 and threesome['silt'] <= 50:
            return 'loam'
        else:
            return 'silt loam'
    elif 15 <= threesome['clay'] <= 20:
        if 52.5 <= threesome['sand']:
            return 'sandy loam'
        elif threesome['silt'] <= 50:
            return 'loam'
        else:
            return 'silt loam'
    elif 10 <= threesome['clay'] <= 15:
        if 70 <= threesome['sand'] <= 85:
            return 'loamy sand'
        if 52.5 <= threesome['sand'] <= 70:
            return 'sandy loam'
        elif 35 <= threesome['silt'] <= 50:
            return 'loam'
        elif 50 <= threesome['silt'] <= 80:
            return 'silt loam'
        else:
            return 'silt'
    elif 0 <= threesome['clay'] <= 10:
        if 85 <= threesome['sand'] <= 100:
            return 'sand'
        elif 70 <= threesome['sand'] <= 85:
            return 'loamy sand'
        elif 42.5 <= threesome['sand'] <= 70 and threesome['silt'] <= 50:
            if 7.5 <= threesome['clay'] <= 10:
                return 'loam'
            else:
                return 'sandy loam'
        elif 42.5 <= threesome['sand'] <= 70 and threesome['silt'] >= 50:
            return 'silt loam'
        elif 50 <= threesome['silt'] <= 80:
            return 'silt loam'
        else:
            return 'silt'


def get_results_df(results_df_name, cols_res_df):
    if os.path.isfile(results_df_name):
        results_df = pd.read_excel(results_df_name)
        file_exist = True
    else:
        results_df = pd.DataFrame(columns=cols_res_df)
        file_exist = False
    return results_df, file_exist


def get_best(net_type, texture_type, index=0):
    file_name = 'results_all_models/{0}/top5-{1}.xlsx'.format(net_type, texture_type)
    if os.path.isfile(file_name):
        df = pd.read_excel(file_name)
        return ast.literal_eval(df['features'][index].replace('array', ''))
    return None


def sd(vec):
    m = sum(vec) / len(vec)
    return ((sum([(item - m) ** 2 for item in vec])) / (len(vec) - 1)) ** 0.5


def RPD(predicted, observed):
    val = sd(observed) / (sum([(predicted[i] - observed[i])**2 for i in range(len(predicted))])/len(observed))**0.5
    return int(val*100)/100


def RPIQ(predicted, observed):
    q = [0, 0.25, 0.5, 0.75, 1]
    quantile = np.quantile(observed, q)
    val = (quantile[3] - quantile[1])/(sum([(predicted[i] - observed[i])**2 for i in range(len(predicted))])/len(observed))**0.5
    return int(val*100)/100


def change_col_name(col_name):
    col_name = col_name.replace('_', ' ')
    col_name = col_name.title()
    col_name += "S" if col_name[-1] == 'M' else 'M'
    return col_name



