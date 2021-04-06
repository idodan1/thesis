import pandas as pd
import pickle

texture_names = ['master sizer', 'hydro meter']

all_data_file = 'soil_data_2020_all data.xlsx'
res_df_name = 'descriptive statistics data frame.xlsx'
all_data_df = pd.read_excel(all_data_file, index_col=0)[2:]
cols = columns=['Statistic', 'Min', 'Max', '1st Quartile', 'Median', '3rd Quartile', 'Mean', 'Max-Min', 'Standard deviation']
res_df = pd.DataFrame(columns=['Statistic', 'Min', 'Max', '1st Quartile', 'Median', '3rd Quartile', 'Mean', 'Standard deviation'])
for texture_name in texture_names:
    with open('texture cols {0}'.format(texture_name), 'rb') as f:
        texture_cols = pickle.load(f)
    for col in texture_cols:
        texture_df = all_data_df[[col]]
        quantile = texture_df[col].quantile([.25, .5, .75])
        row_dict = {'Statistic': col, 'Min': texture_df.min(), 'Max':texture_df.max(), '1st Quartile': quantile[0.25], 'Median':quantile[0.5],
                    '3rd Quartile': quantile[0.75], 'Mean': texture_df.mean(), 'Standard deviation': texture_df.std(), 'Max-Min': texture_df.max()-texture_df.min()}
        row_df = pd.DataFrame.from_dict(row_dict)
        res_df = pd.concat([res_df, row_df])
temp = pd.DataFrame()
for col in cols:
    temp[col] = res_df[col]
res_df = temp
res_df.to_excel(res_df_name, index=False)


