from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap_
import timeit

class Reduction:
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    def pca(self, a, b):
        time_begin = timeit.default_timer()
        pca = PCA(n_components=2)
        pca.fit(a + b)
        result = (pca.transform(a), pca.transform(b))
        print('PCA seconds:', timeit.default_timer() - time_begin)
        return result
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    def tsne(self, a, b):
        time_begin = timeit.default_timer()
        tsne = TSNE().fit_transform(a + b)
        print('t-SNE seconds:', timeit.default_timer() - time_begin)
        return (tsne[0:len(a)], tsne[len(a):])
    
    # https://umap-learn.readthedocs.io/en/latest/api.html#umap.umap_.UMAP.fit_transform
    def umap(self, a, b):
        time_begin = timeit.default_timer()
        umap = umap_.UMAP().fit_transform(a + b)
        print('UMAP seconds:', timeit.default_timer() - time_begin)
        return (umap[0:len(a)], umap[len(a):])