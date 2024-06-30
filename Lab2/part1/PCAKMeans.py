import numpy as np
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors


def load_data():
    words = [
        'computer', 'laptop', 'minicomputers', 'PC', 'software', 'Macbook',
        'king', 'queen', 'monarch', 'prince', 'ruler', 'princes', 'kingdom', 'royal',
        'man', 'woman', 'boy', 'teenager', 'girl', 'robber', 'guy', 'person', 'gentleman',
        'banana', 'pineapple', 'mango', 'papaya', 'coconut', 'potato', 'melon',
        'shanghai', 'HongKong', 'chinese', 'Xiamen', 'beijing', 'Guilin',
        'disease', 'infection', 'cancer', 'illness',
        'twitter', 'facebook', 'chat', 'hashtag', 'link', 'internet',
    ]
    w2v = KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)
    vectors = []
    for w in words:
        vectors.append(w2v[w].reshape(1, 300))
    vectors = np.concatenate(vectors, axis=0)
    return words, vectors


class KernelPCA:
    def __init__(self, n_components=2, kernel='rbf', gamma=None, degree=3, c=1, alpha=1):
        self.eigenvectors = None
        self.eigenvalues = None
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.c = c
        self.alpha = alpha

    def __rbf_kernel(self, X):
        """ 高斯径向基函数核 """
        if self.gamma is None:
            self.gamma = 1 / X.shape[1]
        sq_dists = np.sum(X ** 2, axis=1).reshape(-1, 1) - 2 * X @ X.T + np.sum(X ** 2, axis=1)
        return np.exp(-self.gamma * sq_dists)

    def __poly_kernel(self, X):
        """ 多项式核 """
        return (X @ X.T + self.c) ** self.degree

    def __sigmoid_kernel(self, X):
        """ sigmoid 核 """
        return np.tanh(self.alpha * X @ X.T + self.c)

    def __kernel_function(self, X):
        if self.kernel == 'rbf':
            return self.__rbf_kernel(X)
        elif self.kernel == 'poly':
            return self.__poly_kernel(X)
        elif self.kernel == 'sigmoid':
            return self.__sigmoid_kernel(X)
        else:
            raise ValueError('Invalid kernel type')

    def fit_transform(self, X):
        """
        :param X: shape (n_samples, n_features)
        :return: shape (n_samples, n_components)
        """
        # 中心化核矩阵
        n_samples = X.shape[0]
        K = self.__kernel_function(X)
        one_n = np.ones((n_samples, n_samples)) / n_samples
        K = K - one_n @ K - K @ one_n + one_n @ K @ one_n

        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(K)

        # 选择前 n_components 个特征向量
        idx = eigenvalues.argsort()[::-1]
        self.eigenvalues = eigenvalues[idx][:self.n_components]
        self.eigenvectors = eigenvectors[:, idx][:, :self.n_components]

        # 计算投影
        return K @ self.eigenvectors @ np.diag(1 / np.sqrt(self.eigenvalues))


class KMeans:
    def __init__(self, n_clusters=3, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centers = None

    def __init_centers(self, points):
        n_samples = points.shape[0]
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        return points[indices]

    def __assign_points(self, points):
        # 计算每个点到每个中心的距离
        # points[:, np.newaxis]: n_samples * n_dim -> n_samples * 1 * n_dim
        dists = np.sum((points[:, np.newaxis] - self.centers) ** 2, axis=2)
        return np.argmin(dists, axis=1)

    def __update_centers(self, points):
        labels = self.__assign_points(points)
        new_centers = np.array([points[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return new_centers

    def fit(self, points):
        self.centers = self.__init_centers(points)
        for _ in range(self.max_iter):
            new_centers = self.__update_centers(points)
            if np.allclose(self.centers, new_centers):
                break
            self.centers = new_centers

    def predict(self, points):
        return self.__assign_points(points)


if __name__ == '__main__':
    words, data = load_data()
    pca = KernelPCA(kernel='sigmoid')
    data_pca = pca.fit_transform(data)

    kmeans = KMeans(n_clusters=7)
    kmeans.fit(data_pca)
    clusters = kmeans.predict(data_pca)

    # plot the data

    plt.figure()
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters)
    for i in range(len(words)):
        plt.annotate(words[i], data_pca[i, :])
    plt.title("PB21030794")
    plt.savefig("PCA_KMeans.png")
