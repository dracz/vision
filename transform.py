__author__ = 'dracz'

import numpy as np


def normalize(x, unit_var=False):
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


def whiten(x, tol=1E-18):
    #w, v = pca(x)
    #d = np.diag(1. / np.sqrt(w + tol))
    pass


class PCA:
    def __init__(self, data, epsilon=0.1):
        n, m = data.shape
        self.mean = data.mean(axis=0)
        self.x = data - self.mean

        self.sigma = np.cov(self.x)
        self.u, self.s, v = np.linalg.svd(self.sigma)
        self.x_tilde = None
        self.retaining = None

    def reduce(self, retain=0.9, k=None):
        if k is not None and k > len(self.s):
            raise ValueError("k must be <= number of singular values")
        k = k if k is not None else self.find_k(retain)
        self.retaining = self.s[:k].sum()/self.s.sum()
        self.x_tilde = np.dot(self.u[:,:k].T, self.x)
        return self.x_tilde

    def reconstruct(self, x_tilde):
        return np.dot(self.u[:,:x_tilde.shape[0]], x_tilde) + self.mean

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

    def diag(self, epsilon=0.1):
        return np.diag(1/np.sqrt(np.diag(self.s)) + epsilon)

    def whiten_pca(self, epsilon=0.1):
        return np.dot(np.dot(self.diag(epsilon), self.u.T), self.x)

    def whiten_zca(self, epsilon=0.1):
        np.dot(np.dot(np.dot(self.u, self.diag(epsilon)), self.u.T), self.x)


def test_pca():
    from PIL import Image
    from images import sample_patches
    from lfw import lfwc_paths

    sh, m = (64, 64), 500
    n_faces = 20
    levels = np.arange(0.95, 0.1, -0.05)

    print("loading {} {} images...".format(m, sh))
    x = np.asarray([p.flatten() for p in list(sample_patches(lfwc_paths(), sh, m))]).T

    print("performing PCA...")
    pca = PCA(x)

    imgs = []

    img = Image.new("L", (sh[0]*(len(levels) + 1), sh[1]*n_faces))

    for i in range(n_faces):
        img.paste(Image.fromarray(x[:,i].reshape(sh)), (0, i*sh[1]))

    for j, level in enumerate(levels):
        print("retaining {} variance...".format(level))
        reduced = pca.reduce(retain=level)
        x_hat = pca.reconstruct(reduced)

        for i in range(n_faces):
            img.paste(Image.fromarray(x_hat[:,i].reshape(sh)), ((j+1)*sh[0], i*sh[1]))

    img.show()
    img.save("pca_faces.png")


if __name__ == "__main__":
    test_pca()