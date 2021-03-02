from RandomModel import *
import pandas as pd

all_data_file = 'soil_data_2020_all data.xlsx'
all_data_df = pd.read_excel(all_data_file, index_col=0)[2:]

model = RandomModel()

