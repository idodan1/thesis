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


def find_k(data, data_cols, data_names):
    for i in range(len(cols_type)):
        data = all_data_df[cols_type[i]]

        gap, reference_inertia, ondata_inertia = compute_gap(KMeans(), data, k_max)
        line1, = plt.plot(range(1, k_max + 1), reference_inertia,
                          '-o', label='reference')
        line2, = plt.plot(range(1, k_max + 1), ondata_inertia,
                          '-o', label='data')
        plt.xlabel('k')
        plt.ylabel('log(inertia)')
        plt.legend((line1, line2), ('reference', 'data'))
        plt.title('{0} data compered to reference'.format(texture_names[i]))
        plt.savefig('gap statistics/{0} data compered to reference'.format(texture_names[i]))
        plt.show()
        plt.clf()

        plt.plot(range(1, k_max + 1), gap, '-o')
        plt.title('gap plot for {0}'.format(texture_names[i]))
        plt.ylabel('gap')
        plt.xlabel('k')
        plt.savefig('gap statistics/gap plot for {0}'.format(texture_names[i]))
        plt.show()
        plt.clf()


def plot_k_clusters(k, data, data_cols, data_type):
    curr_data = data[data_cols]
    kmeans = KMeans(init='random',
                    n_clusters=k,
                    n_init=10, max_iter=1000, random_state=42)
    kmeans.fit(curr_data)
    centroids = kmeans.cluster_centers_

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
    for i in range(3):
        axes[i].scatter(data[data_cols[i - 1]], data[data_cols[i - 2]], c=kmeans.labels_.astype(float), s=50, alpha=0.5,
                        label='')
        axes[i].scatter(centroids[:, i - 1], centroids[:, i - 2], c='red', s=50, label="centroids")
        axes[i].set_xlabel(data_cols[i - 1])
        axes[i].set_ylabel(data_cols[i - 2])
        axes[i].set_title("{0} to {1}".format(data_cols[i - 1], data_cols[i - 2]))
        axes[i].legend()
    plt.show()
    plt.savefig('gap statistics/{0} division to clusters'.format(data_type))
    return kmeans


def save_train_test(kmeans, data, data_type, data_cols, k):
    curr_data = data[data_cols]
    curr_data['cluster'] = kmeans.predict(curr_data)
    train, test = divide_to_test_n_train(curr_data, k)

    with open('train nums {0}'.format(data_type), 'wb') as f:
        pickle.dump(train, f)
    with open('test nums {0}'.format(data_type), 'wb') as f:
        pickle.dump(test, f)


def create_reference_division(k, data_type):
    test = np.random.choice(list(range(1, 64)), k, replace=False)
    train = set(range(1, 64)) - set(test)
    with open('train nums random {0}'.format(data_type), 'wb') as f:
        pickle.dump(train, f)
    with open('test nums random {0}'.format(data_type), 'wb') as f:
        pickle.dump(test, f)


# with open('train_nums', 'rb') as f:
#     train_nums = pickle.load(f)
all_data_file = 'soil_data_2020_all data.xlsx'
start_index = 2
last_index = start_index + 63
k_max = 30
all_data_df = pd.read_excel(all_data_file, index_col=0)[start_index: last_index]
with open('texture cols master sizer', 'rb') as f:
    texture_master_cols = pickle.load(f)
with open('texture cols hydro meter', 'rb') as f:
    texture_hydro_cols = pickle.load(f)

cols_type = [texture_hydro_cols, texture_master_cols]
texture_names = ['hydro meter', 'master sizer']
chosen_k = {'hydro meter': 8, 'master sizer': 6}

# find_k(all_data_df, cols_type, texture_names)
# for i in range(len(texture_names)):
#     kmeans = plot_k_clusters(chosen_k[texture_names[i]], all_data_df, cols_type[i], texture_names[i])
#     save_train_test(kmeans, all_data_df, texture_names[i], cols_type[i], chosen_k[texture_names[i]])
for i in range(len(texture_names)):
    create_reference_division(chosen_k[texture_names[i]], texture_names[i])












# with open('train_nums_net', 'wb') as f:
#     pickle.dump(train, f)
# with open('val_nums_net', 'wb') as f:
#     pickle.dump(val, f)
# train, val = divide_to_test_n_train(data, chosen_k)
# print(len(test))
# print(val)