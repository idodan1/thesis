import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.datasets import make_blobs
from scipy.spatial import distance
from scipy.ndimage.interpolation import shift
import scipy
from IPython.display import Image
import seaborn as sns
from sklearn.cluster import KMeans


def compute_inertia(a, X):
    if isinstance(X, pd.DataFrame):
        X = np.array([list(X[i:i+1].values[0]) for i in range(len(X))])
    W = [np.mean(pairwise_distances(X[a == c, :])) for c in np.unique(a)]
    return np.mean(W)


def compute_gap(clustering, data, k_max=15, n_references=5):
    reference = np.random.rand(*data.shape)
    reference_inertia = []
    for k in range(1, k_max + 1):
        local_inertia = []
        for _ in range(n_references):
            clustering.n_clusters = k
            assignments = clustering.fit_predict(reference)
            local_inertia.append(compute_inertia(assignments, reference))
        reference_inertia.append(np.mean(local_inertia))

    ondata_inertia = []
    for k in range(1, k_max + 1):
        clustering.n_clusters = k
        assignments = clustering.fit_predict(data)
        ondata_inertia.append(compute_inertia(assignments, data))

    gap = np.log(reference_inertia) - np.log(ondata_inertia)
    return gap, np.log(reference_inertia), np.log(ondata_inertia)


def divide_to_test_n_train(data, num_of_clusters):
    train = []
    test = []
    for i in range(num_of_clusters):
        df_p = data.loc[data['cluster'] == i]
        the_rest = list(df_p.index)
        r = np.random.choice(the_rest, 1)[0]
        the_rest.remove(r)
        test.append(r)
        train.extend(the_rest)
    return train, test


# with open('train_nums', 'rb') as f:
#     train_nums = pickle.load(f)
all_data_file = 'soil_data_2020_all data.xlsx'
start_index = 2
last_index = start_index + 63
k_max = 30
all_data_df = pd.read_excel(all_data_file, index_col=0)[start_index: last_index]
with open('texture_master_cols', 'rb') as f:
    data_cols = pickle.load(f)
data = all_data_df[data_cols]
# data = data.ix[train_nums]

# gap, reference_inertia, ondata_inertia = compute_gap(KMeans(), data, k_max)
# line1, = plt.plot(range(1, k_max+1), reference_inertia,
#          '-o', label='reference')
# line2, = plt.plot(range(1, k_max+1), ondata_inertia,
#          '-o', label='data')
# plt.xlabel('k')
# plt.ylabel('log(inertia)')
# plt.legend((line1, line2), ('reference', 'data'))
# plt.show()
#
# plt.plot(range(1, k_max+1), gap, '-o')
# plt.ylabel('gap')
# plt.xlabel('k')
# plt.show()


chosen_k = 6
kmeans = KMeans(init='random',
                n_clusters=chosen_k,
                n_init=10, max_iter=1000, random_state=42)
kmeans.fit(data)
centroids = kmeans.cluster_centers_

fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(20,5))
for i in range(3):
    axes[i].scatter(data[data_cols[i-1]], data[data_cols[i-2]], c=kmeans.labels_.astype(float), s=50, alpha=0.5, label='g')
    axes[i].scatter(centroids[:, i-1], centroids[:, i-2], c='red', s=50, label="centroids")
    axes[i].set_xlabel(data_cols[i-1])
    axes[i].set_ylabel(data_cols[i-2])
    axes[i].set_title("{0} to {1}".format(data_cols[i-1], data_cols[i-2]))
    axes[i].legend()

plt.show()


data['cluster'] = kmeans.predict(data)
# # print(data['cluster'])
# # print(data['cluster'].value_counts())
train, test = divide_to_test_n_train(data, chosen_k)
# train, val = divide_to_test_n_train(data, chosen_k)
print(train)
print(len(train))
print(test)
print(len(test))
# print(val)
with open('train_nums', 'wb') as f:
    pickle.dump(train, f)
with open('test_nums', 'wb') as f:
    pickle.dump(test, f)
# with open('train_nums_net', 'wb') as f:
#     pickle.dump(train, f)
# with open('val_nums_net', 'wb') as f:
#     pickle.dump(val, f)

