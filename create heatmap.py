import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

df = pd.read_excel('soil_data_2020_all data.xlsx')
df = df[['ECaV_2020', 'ECaH_2020', 'MSaV_2020', 'MSaH_2020', 'ECaV_2019', 'ECaH_2019', 'MSaV_2019', 'MSaH_2019',
         'ECaV_2018', 'ECaH_2018', 'MSaV_2018', 'MSaH_2018', 'ECaH_man', 'MSaH_man', 'ECaV_man', 'MSaV_man']]
df = df[2:]
df = df.astype(float)
fig = plt.figure(figsize=(20, 20))
title_font_size = 35
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_title('2020', fontsize=title_font_size)
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_title('2019', fontsize=title_font_size)
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_title('2018', fontsize=title_font_size)
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_title('2020 man', fontsize=title_font_size)
font_size = 28
label_font_size = 28
ticks = ['ECaV', 'ECaH', 'MSaV', 'MSaH']
sb.set(font_scale=2.5)
res = sb.heatmap(data=df[['ECaV_2020', 'ECaH_2020', 'MSaV_2020', 'MSaH_2020']].corr(), ax=ax1, square=True, cbar_kws={'shrink': .3}, annot=True, annot_kws={'fontsize': font_size})
res.set_xticklabels(ticks, fontsize=label_font_size)
res.set_yticklabels(ticks, fontsize=label_font_size)
res = sb.heatmap(data=df[['ECaV_2019', 'ECaH_2019', 'MSaV_2019', 'MSaH_2019']].corr(), ax=ax2, square=True, cbar_kws={'shrink': .3}, annot=True, annot_kws={'fontsize': font_size})
res.set_xticklabels(ticks, fontsize=label_font_size)
res.set_yticklabels(ticks, fontsize=label_font_size)
res = sb.heatmap(data=df[['ECaV_2018', 'ECaH_2018', 'MSaV_2018', 'MSaH_2018']].corr(), ax=ax3, square=True, cbar_kws={'shrink': .3}, annot=True, annot_kws={'fontsize': font_size})
res.set_xticklabels(ticks, fontsize=label_font_size)
res.set_yticklabels(ticks, fontsize=label_font_size)
res = sb.heatmap(data=df[['ECaH_man', 'MSaH_man', 'ECaV_man', 'MSaV_man']].corr(), ax=ax4, square=True, cbar_kws={'shrink': .3}, annot=True, annot_kws={'fontsize': font_size})
res.set_xticklabels(ticks, fontsize=label_font_size)
res.set_yticklabels(ticks, fontsize=label_font_size)
plt.savefig('4 heat maps')
# plt.show()
plt.clf()
sb.heatmap(data=df[['ECaV_2020', 'ECaV_2019', 'ECaV_man', 'ECaV_2018']].corr(), annot=True)

# plt.show()


