import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score


# sklearn.KMeans uses Euclidean distance as similarity metric and is not changeable
def run_k_means(df: pd.DataFrame, number_of_clusters: int):
    print('Running k_means with number of clusters: ', number_of_clusters)
    k_means = KMeans(n_clusters=number_of_clusters, random_state=0).fit(df)

    score = silhouette_score(df, k_means.labels_, metric='euclidean')
    print('Silhouette Score: %.5f' % score)


# sklearn.DBSCAN uses Euclidean distance as default but can be changed to cosine
def run_dbscan(df: pd.DataFrame, epsilon, minimum_samples):
    print('Running DBScAN with epsilon: ', epsilon, ' minimum samples: ', minimum_samples)
    dbscan = DBSCAN(eps=epsilon, min_samples=minimum_samples).fit(df)

    score = silhouette_score(df, dbscan.labels_, metric='euclidean')
    print('Silhouette Score: %.5f' % score)
