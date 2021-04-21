import pandas as pd
import ast


def get_parameters(texture_name, index):
    df = pd.read_excel('results_all_models/super model/res df {0}.xlsx'.format(texture_name))
    values = {col: df[col][index] for col in df.columns}
    values['features'] = ast.literal_eval(values['features'])
    values['num of neurons'] = ast.literal_eval(values['num of neurons'])
    values['activation nums'] = ast.literal_eval(values['activation nums'])
    print(df.columns)
    return values

def get_parameters_for_conv_img(texture_name, index):
    df = pd.read_excel('results_all_models/conv_img/res df {0} division:  random.xlsx'.format(texture_name))
    values = {col: df[col][index] for col in df.columns}
    return values

textures = ['hydro meter', 'master sizer']
for t in textures:
    get_parameters_for_conv_img(t, 0)