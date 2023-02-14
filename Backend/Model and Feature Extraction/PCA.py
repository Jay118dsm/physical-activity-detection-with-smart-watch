from sklearn.decomposition import PCA
import Data_Input
import read_features


def get_pca_3(X):
    pca = PCA(n_components=99)
    pca_reduced_X = pca.fit_transform(X)
    print("Explained_variance_ratio: ", pca.explained_variance_ratio_)
    print(sum(pca.explained_variance_ratio_))
    return pca_reduced_X





X = read_features.return_features_whole_UCI_10299_items()
pca_matrix = get_pca_3(X)
def return_PCA3():
    return pca_matrix