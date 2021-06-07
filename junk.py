# import pandas as pd
# import ast
#
#
# def get_parameters(texture_name, index):
#     df = pd.read_excel('results_all_models/super model/res df {0}.xlsx'.format(texture_name))
#     values = {col: df[col][index] for col in df.columns}
#     values['features'] = ast.literal_eval(values['features'])
#     values['num of neurons'] = ast.literal_eval(values['num of neurons'])
#     values['activation nums'] = ast.literal_eval(values['activation nums'])
#     print(df.columns)
#     return values
#
# def get_parameters_for_conv_img(texture_name, index):
#     df = pd.read_excel('results_all_models/conv_img/res df {0} division:  random.xlsx'.format(texture_name))
#     values = {col: df[col][index] for col in df.columns}
#     return values
#
# textures = ['hydro meter', 'master sizer']
# for t in textures:
#     get_parameters_for_conv_img(t, 0)
#

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def open_image(adress):
    img = cv.imread(adress)
    img_convert = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img_convert

ad = "images/1_a_1.JPG"
start_index = 512
image = open_image(ad)
# plt.imshow(image)
# plt.show()
image = np.asarray(image, dtype=np.int8)
for j in range(20):
    for i in range(2000):
        image[start_index+j][start_index+i] = [0, 0, 0]
for j in range(20):
    for i in range(2000):
        image[start_index+j+2000][start_index+i] = [0, 0, 0]
for j in range(20):
    for i in range(2000):
        image[start_index+i][start_index+j] = [0, 0, 0]
for j in range(20):
    for i in range(2000):
        image[start_index+i][start_index+j+2000] = [0, 0, 0]
plt.imshow(image.astype('uint8'))
plt.show()