
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from images import sample_patches
from lfw import lfwc_paths

__author__ = 'dracz'


def normalize(x):
    """
    Normalize (m by w by h) 3d ndarray x, to have zero mean and optionally unit variance
    Truncate to +/- 3 standard deviations, scale to range [0., 1.]
    :return: normalized 3d ndarray
    """
    x -= x.mean(axis=0)
    std = 3 * x.std()
    x[x < -std] = -std
    x[x > std] = std
    x /= std
    return (x + 1) * 0.5


epsilon = 10.0**-12


class PCA:
    def __init__(self, data):
        n, m = data.shape
        self.x = data - data.mean(axis=0)
        self.sigma = np.dot(self.x, self.x.T) / self.x.shape[1]
        self.u, self.s, self.v = np.linalg.svd(self.sigma)

    def reduce(self, x, retain=0.9, k=None):
        """return reduced dimension version of vector x"""
        if k is not None and k > len(self.s):
            raise ValueError("k must be <= number of singular values")
        k = k if k is not None else self.find_k(retain)
        mean = x.mean(axis=0)
        return np.dot(self.u[:, :k].T, x - mean), mean

    def reconstruct(self, x_tilde, mean=0):
        u_reduced = self.u[:, :x_tilde.shape[0]]
        dp = np.dot(u_reduced, x_tilde)
        return dp + mean

    def retaining(self, k):
        return self.s[:k].sum()/self.s.sum()

    def find_k(self, retain=0.99):
        if retain >= 1:
            return len(self.s)
        tot = self.s.sum()
        s_sum = 0.0
        for i in range(len(self.s)):
            s_sum += self.s[i]
            if s_sum / tot >= retain:
                return i+1
        raise ValueError("sum of singular values < {}".format(retain))

    def x_whiten_pca(self, x, eps=epsilon):
        d = np.diag(1. / np.sqrt(self.s + eps))
        return np.dot(np.dot(d, self.u.T), x)

    def x_whiten_zca(self, eps=epsilon):
        return np.dot(self.u, self.whiten_pca(eps))


def test_pca_input(sh, m):
    print("loading {} {} images...".format(m, sh))
    return np.asarray([p.flatten() for p in list(sample_patches(lfwc_paths(), sh, m))]).T


def pca_faces_image(sh=(64, 64), m=500, n_faces=20, k_vals=None, img_out="pca_faces.png", test_in_sample=True):
    """generate an image of faces reduced and reconstructed retaining various top-k components"""

    tot = m if test_in_sample else m + n_faces
    data = test_pca_input(sh, tot)
    x = data[:, :m]

    test_x = data[:, :n_faces] if test_in_sample else data[:, m:]

    pca = PCA(x)
    retain = [0.99, 0.95] + list(np.arange(0.90, 0.1, -0.1))
    k_vals = k_vals if k_vals is not None else [pca.find_k(p) for p in retain]

    img = Image.new("L", (sh[0]*(len(k_vals) + 1), sh[1]*n_faces))

    for i in range(test_x.shape[1]):
        img.paste(Image.fromarray(x[:, i].reshape(sh)), (0, i*sh[1]))

        for j, k in enumerate(sorted(k_vals, reverse=True)):
            x_tilde, mean = pca.reduce(test_x[:, i], k=k)
            x_hat = pca.reconstruct(x_tilde, mean=mean).reshape(sh)
            img.paste(Image.fromarray(x_hat), ((j+1)*sh[0], i*sh[1]))
    img.show()
    img.save(img_out)


def test_pca(sh=(12, 12), m=500, retain=0.9):
    # should show that the covariance matrix of x_tilde
    # is a diagonal matrix with descending values
    x = test_pca_input(sh, m)
    pca = PCA(x)
    xt = pca.reduce(retain=retain)
    cov = np.dot(xt, xt.T)
    plt.imshow(cov)
    plt.show()


def test_pca_white(sh=(12, 12), m=500, eps=0.1):
    # should show that the covariance matrix of whitening matrix
    # is a diagonal matrix with descending values
    x = test_pca_input(sh, m)
    pca = PCA(x)
    w = pca.x_whiten_pca(eps=eps)
    cv = np.dot(w, w.T)
    plt.imshow(cv)
    plt.show()


if __name__ == "__main__":
    test_pca()
    test_pca_white()
    pca_faces_image(sh=(64, 64), m=100000, n_faces=20)