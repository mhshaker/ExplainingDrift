from sklearn.cluster import KMeans

class Clustering:

    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    def kmeans(self, points, n_clusters=2):
        results = KMeans(n_clusters=n_clusters).fit_predict(points)
        clusters = []
        for i in range(n_clusters):
            clusters.append([])
        for i, cluster in enumerate(results):
            clusters[cluster].append(points[i])
        return clusters

    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    def kmeans(self, points, indexes, n_clusters=2):
        results = KMeans(n_clusters=n_clusters).fit_predict(points)
        clusters = []
        for i in range(n_clusters):
            clusters.append([])
        for i, cluster in enumerate(results):
            clusters[cluster].append(indexes[i])
        return clusters