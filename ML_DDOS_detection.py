### SIADS 696 : MILESTONE PROJECT 2 - DDOS DETECTION UNSUPERVISED LEARNING
### MANISH JAKHI
###-----------------------------------------------------------------------------------------------------------###
## INITIALIZATIONS

import pandas as pd
pd.set_option('display.max_columns', 35)
pd.set_option('display.width', 1000)
import numpy as np
from sklearn.model_selection import train_test_split
from time import time
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from sklearn.cluster import DBSCAN
import seaborn as sns


###-----------------------------------------------------------------------------------------------------------###

# rear csv files into a df
df_ddos = pd.read_csv ('CSVs/ddos_traffic_model_ready.csv')
# print(df_ddos)

df_normal = pd.read_csv ('CSVs/normal_traffic_model_ready.csv')
# print(df_normal)

# concatenating df_ddos and df_normal along rows
df_comb = pd.concat([df_ddos, df_normal], axis=0)
# df_comb = df_comb.sample(frac = 1)

## transforming CATEGORY label column to binary,  Benign traffic, 0 & DDOS traffic = 1
# df_comb['CATEGORY'] = np.where(df_comb['CATEGORY']== 'Benign Traffic', 0, 1)

###-----------------------------------------------------------------------------------------------------------###

## Numerical Mapping for categories
CATEGORY_MAP = {'Slow HTTP GET':1, 'Slow Read':2, 'Slow Post':3, 'GoldenEye':4, 'Hulk':5, 'Benign Traffic':0}
print(df_comb['CATEGORY'].value_counts())
CATEGORY_MAP0 = {'DDOS Traffic':1, 'Benign Traffic':0}
CATEGORY_MAP1 = {'Slow HTTP GET':1, 'Slow Read':2, 'Slow Post':3, 'GoldenEye':4, 'Hulk':5, 'Benign Traffic':0}


CATEGORY_MAP2 = {'Slow HTTP GET':0, 'Slow Read':0, 'Slow Post':0, 'GoldenEye':0, 'Hulk':0, 'Benign Traffic':1}
df_comb['CATEGORY2'] = df_comb['CATEGORY'].map(CATEGORY_MAP2)

print(df_comb.head(10))
print(df_comb.columns)

###-----------------------------------------------------------------------------------------------------------###

## Feature Extraction - selecting columns that need to be used as input to Unsupervised learning model
# X= df_comb[['Packets', 'Bytes', 'Packets_A-to-B(TX_pkts)', 'Bytes_A-to-B(TX_bytes)',
#             'Packets_B-to-A(RX_pkts)', 'Bytes_B-to-A(RX_bytes)',
#             'Bits/s A-to-B (TX_bits/s)', 'Bits/s B-to-A (RX_bits/s)', 'tot_kbps',
#             'Pktrate', 'Flow Duration', 'Idle Timeout', 'Hard Timeout',
#             'Pktrate nsec', 'flags']]
# X= df_comb[['Packets_A-to-B(TX_pkts)', 'Bytes_A-to-B(TX_bytes)',
#             'Packets_B-to-A(RX_pkts)', 'Bytes_B-to-A(RX_bytes)','tot_kbps',
#             'Pktrate', 'Flow Duration', 'Idle Timeout', 'flags']]

#%%
X = df_comb.drop(labels=['Address-A', 'Port-A', 'Address-B', 'Port-B', 'StreamID',
                         'PerctageFiltered', 'CATEGORY', 'Relative_Start',
                         'TotalPackets', 'Syn', 'Acknowledgment', 'Fin', 'Reset',
                         'Length', 'TCP Segment Len', 'Push', 'CATEGORY2',
                         'Duration', 'Header Length','Packets', 'Bytes', 'Bytes In Flight',
                         'Bits/s A-to-B (TX_bits/s)', 'Bits/s B-to-A (RX_bits/s)',
                         'Pktrate', 'Flow Duration','Pktrate nsec', 'Byte nsec'], axis=1)
print(X.columns)

y = df_comb['CATEGORY2']
print(y.value_counts())


print(f"# labels: {np.unique(y).size}; # samples: {X.shape[0]}; # features {X.columns.size}")

###-----------------------------------------------------------------------------------------------------------###

# using the train test split function
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=104, test_size=0.3, shuffle=True)

###-----------------------------------------------------------------------------------------------------------###

def plot_labelled_scatter(X, y, class_labels, title = ""):
    num_labels = len(class_labels)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    marker_array = ['o', '^', '*']
    color_array = ['#FFFF00', '#00AAFF', '#000000', '#FF00AA']
    cmap_bold = ListedColormap(color_array)
    bnorm = BoundaryNorm(np.arange(0, num_labels + 1, 1), ncolors=num_labels)
    plt.figure()

    plt.scatter(X[:, 0], X[:, 1], s=65, c=y, cmap=cmap_bold, norm = bnorm, alpha = 0.40, edgecolor='black', lw = 1)

    plt.title(title)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    h = []
    for c in range(0, num_labels):
        h.append(mpatches.Patch(color=color_array[c], label=class_labels[c]))
    plt.legend(handles=h)

    plt.show()

###-----------------------------------------------------------------------------------------------------------###

def benchmark(model, name, data, labels):
    """Benchmark to evaluate the KMeans initialization methods.

    Parameters
    ----------
    model : model instance
        A :class:`~sklearn.cluster.KMeans` instance with the initialization
        already set.
    name : str
        Name given to the strategy. It will be used to show the results in a
        table.
    data : ndarray of shape (n_samples, n_features)
        The data to cluster.
    labels : ndarray of shape (n_samples,)
        The labels used to compute the clustering metrics which requires some
        supervision.
    """
    t0 = time()
    estimator = make_pipeline(StandardScaler(), model).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.precision_score,
        metrics.recall_score,
        metrics.f1_score,
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # Show the results
    formatter_result = (
        "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
    )
    print(formatter_result.format(*results))

    # y_pred = estimator[-1].labels_
    # print(classification_report(y, y_pred, target_names=list(CATEGORY_MAP2.keys())))
    # print(results)

print(82 * "_")
print("init\t\ttime\tinertia\tPrecision\tRecall\tF1\t\tHomo\tcompl\tv-meas\tARI\t\tAMI")
kmeans = MiniBatchKMeans(n_clusters=2, n_init="auto", random_state=0,batch_size=6, max_iter=10)
benchmark(model=kmeans, name="MiniBatch", data=X, labels=y)

kmeans = KMeans(init="k-means++", n_clusters=2, n_init=4, random_state=0)
benchmark(model=kmeans, name="k-means++", data=X, labels=y)

kmeans = KMeans(init="random", n_clusters=2, n_init=4, random_state=0)
benchmark(model=kmeans, name="random", data=X, labels=y)

pca = PCA(n_components=2).fit(X)
kmeans = KMeans(init=pca.components_, n_clusters=2, n_init=1)
benchmark(model=kmeans, name="PCA-based", data=X, labels=y)

print(82 * "_")

###-----------------------------------------------------------------------------------------------------------###



###-----------------------------------------------------------------------------------------------------------###

## PCA Analysis / DBSCAN clustering
X_DDOS_normalized = StandardScaler().fit(X).transform(X)

df_comb['CATEGORY1'] = df_comb['CATEGORY'].map(CATEGORY_MAP1)

y_cat1 = df_comb['CATEGORY1']
print(y_cat1.value_counts())

# Declaring Model
# dbscan = DBSCAN()

# Fitting
# dbscan.fit(X_DDOS_normalized)

# Transforming Using PCA
pca = PCA(n_components = 2).fit(X_DDOS_normalized)
X_pca = pca.transform(X_DDOS_normalized)


print(X_DDOS_normalized.shape ,X_pca.shape)
print(y_cat1.nunique())

# Plot based on Class
# for i in range(0, X_pca.shape[0]):
#     if dbscan.labels_[i] == 0:
#         c1 = plt.scatter(X_pca[i, 0], X_pca[i, 1], c='r', marker='+')
#     elif dbscan.labels_[i] == 1:
#         c2 = plt.scatter(X_pca[i, 0], X_pca[i, 1], c='g', marker='o')
#     elif dbscan.labels_[i] == -1:
#         c3 = plt.scatter(X_pca[i, 0], X_pca[i, 1], c='b', marker='*')

# plt.legend([c1, c2, c3], ['Cluster 1', 'Cluster 2', 'Noise'])
# plt.title('DBSCAN finds 2 clusters and Noise')
# plt.show()

plt.figure(figsize=(16,16))
g1 = sns.scatterplot(data = X_pca, x=X_pca[:, 0], y=X_pca[:, 1],
                     s= 100, c=y_cat1, hue=y_cat1, cmap='Spectral',alpha=0.7)
plt.title('Visualizing DDoS attacks through PCA', fontsize=24);
plt.show()


## T-SNE analysis

# Defining and Fitting Model
# tsne_mdl = TSNE(random_state = 42, n_components=2,verbose=0, perplexity=40, n_iter=600).fit_transform(X_pca)
#
# # transformed = model.fit_transform(X)
# #
# ## Plotting 2d t-Sne
# g1 = sns.scatterplot(data = tsne_mdl, x =tsne_mdl[:, 0], y =tsne_mdl[:, 1],
#                      s= 100, c=y_cat1, hue = y_cat1, cmap='Spectral',alpha=0.7)
# plt.title('Visualizing DDoS attacks through t-SNE', fontsize=24);
# plt.show()

###-----------------------------------------------------------------------------------------------------------###

## Plotting the magnitude of each feature value for the first two principal components

# def plot_DDOS_pca(pca, top_k=2):
#     fig, ax = plt.subplots(figsize=(10, 6))
#     plt.imshow(pca.components_[0:top_k], interpolation='none', cmap='plasma')
#     feature_names = list(X.columns)
#     plt.xticks(np.arange(-0., len(feature_names), 1), feature_names, rotation=90, fontsize=12)
#     plt.yticks(np.arange(0., 2, 1), ['First PC', 'Second PC'], fontsize=16)
#     plt.colorbar()
#     plt.show()
#
# plot_DDOS_pca(pca)

###-----------------------------------------------------------------------------------------------------------###

