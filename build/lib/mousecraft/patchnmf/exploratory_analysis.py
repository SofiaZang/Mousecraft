from sklearn.decomposition import PCA
from rastermap import Rastermap, utils 
from openTSNE import TSNE
from umap import UMAP
from sklearn.cluster import KMeans

from scipy.stats import zscore

def fit_pca(data):
    print('fitting PCA...')
    pca=PCA(n_components=200)
    pca_model = pca.fit_transform(data) #Fit the model with X and apply the dimensionality reduction on X.
    pca_emb = pca.components_ #embedding 
    var_exp = pca.explained_variance_ratio_
    # pc1 = pca_model[:,0]
    # pc2=pca_model[:,1]
    # pc3=pca_model[:,2]
    
    return pca_emb, var_exp

def fit_tsne(data):
    print('fitting tSNE...')
    # default openTSNE params
    tsne = TSNE(
        perplexity=30,
        initialization="pca",
        metric="euclidean",
        n_jobs=8,
        random_state=3,
    )

    tsne_emb = tsne.fit(data)
    return tsne_emb

def fit_tsne_1d(data):
    print('fitting 1d-tSNE...')
    # default openTSNE params
    tsne = TSNE(
        n_components=1,
        perplexity=30,
        initialization="pca",
        metric="euclidean",
        n_jobs=8,
        random_state=3,
    )

    tsne_emb = tsne.fit(data)
    return tsne_emb

def fit_umap_3d(data):
    print('fitting UMAP...')
    umap3d = UMAP(n_components=3, init='random', random_state=0)
    umap_emb_obj = umap3d.fit(data.T)
    umap_emb = umap_emb_obj.embedding_
    return umap_emb


def fit_rmap_1d(data):
    print('fitting rmap...')
    rmap_model = Rastermap(n_PCs=200).fit(data) #opposite convention than tSNE and PCA implementations
    rmap_emb_cell = rmap_model.embedding
    isort = rmap_model.isort
    return rmap_emb_cell 

# rastermap_model = Rastermap(n_PCs=200, n_clusters=100, locality=0.75, time_lag_window=5).fit(z_scored_data)
# rmap_model = rastermap_model.embedding 
# isort = rastermap_model.isort

def fit_kmeans(data, n_clusters=8):
    print('fitting kMeans...')
    kmeans = KMeans(n_clusters=n_clusters)
    clustered = kmeans.fit(data)
    labels = clustered.labels_
    return labels

def get_pixel_centroids(tiff, num_frames, x_axis, y_axis):
    # Assuming tiff shape is (num_frames, y_axis_FOV, x_axis_FOV)
    tiff.reshape(num_frames, x_axis, y_axis)
    centroids = np.zeros((y_axis, x_axis, 2))

    for y in range(y_axis):
        for x in range(x_axis):
            centroids[y, x] = [y, x]

    return centroids