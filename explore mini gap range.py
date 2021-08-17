import pickle
import numpy as np
import pandas as pd


df = pd.DataFrame(columns=['model', 'data set', 'mean gap sand', 'mean gap silt', 'mean gap clay'])
for model in ['Conv', 'conv_img']:
    for i in range(0, 4, 2):
        file_name = 'mini_image_range-{0} {1}'
        with open(file_name.format(model, i), 'rb') as f:
            range_list_test = pickle.load(f)
        with open(file_name.format(model, i+1), 'rb') as f:
            range_list_train = pickle.load(f)

        for lst in [range_list_test, range_list_train]:
            gaps = np.array([[abs(l[i]-l[i+1]) for i in range(0, 6, 2)] for l in lst])
            gaps_mean = [np.mean(gaps[:, i]) for i in range(3)]












