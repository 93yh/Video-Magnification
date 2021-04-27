from sklearn.decomposition import PCA


def apply_pca(time_serie):
    pca = PCA()
    eigen_vectors = pca.fit_transform(time_serie.T)
    print("reduced matrix shape: ", pca.components_.shape)
    print("eigenvectors shape: ", eigen_vectors.shape, '\n')
    return eigen_vectors.T, pca.singular_values_, pca.components_.T
