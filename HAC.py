import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
def perfrom_clustering(X,connectivity,title,num_clusters=3,linkage='ward'):
    plt.figure()
    model = AgglomerativeClustering(linkage=linkage,
            connectivity=connectivity,n_clusters=num_clusters)
    model.fit(X)
    labels = model.labels_
    markers = '.vx'
    for i, marker in zip(range(num_clusters), markers):
        plt.scatter(X[labels == i, 0], X[labels == i, 1], s=50,
                    marker=marker, color='k', facecolor='none')
    plt.title(title)

    def get_spiral(t, noise_amplitude=0.5):
        r = t
        x = r * np.cos(t)
        y = r * np.sin(t)
        return add_noise(x, y, noise_amplitude)

    def add_noise(x, y, amplitude):
        X = np.concatenate((x, y))
        X += amplitude * np.random.randn(2, X.shape[1])
        return X.T

    def get_rose(t, noise_amplitude=0.02):
        k = 5
        r = np.cos(k * t) + 0.25
        x = r * np.cos(t)
        y = r * np.sin(t)
        return add_noise(x, y, noise_amplitude)

    def get_hypotrochoid(t, noise_amplitude=0):
        a, b, h = 10.0, 2.0, 4.0
        x = (a - b) * np.cos(t) + h * np.cos((a - b) / b * t)
        y = (a - b) * np.sin(t) - h * np.sin((a - b) / b * t)
        return add_noise(x, y, 0)

    if __name__ == '__main__':
        n_samples = 500
        np.random.seed(2)
        t = 2.5 * np.pi * (1 + 2 * np.random.rand(1, n_samples))
        X = get_spiral(t)
        connectivity = None
        perfrom_clustering(X, connectivity, 'No connectivity')
        connectivity = kneighbors_graph(X, 10, include_self=False)
        perfrom_clustering(X, connectivity, 'K-Neighbora connectivity')
        plt.show()