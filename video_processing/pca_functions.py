from sklearn.decomposition import PCA


def apply_pca(time_serie):
    pca = PCA()
    components = pca.fit_transform(time_serie)
    print("Shape da matriz reduzida: ", components.shape)
    print("Shape dos auto vetores: ", pca.components_.shape, '\n')
    return pca.components_, pca.singular_values_, components
