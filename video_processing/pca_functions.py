from sklearn.decomposition import PCA


def apply_pca(time_serie):
    pca = PCA()
    components = pca.fit_transform(time_serie)
    print("reduced matrix shape: ", components.shape)
    print("eigenvectors shape: ", pca.components_.shape, '\n')
    return pca.components_, pca.singular_values_, components
