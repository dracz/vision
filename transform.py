
import os
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

from images import sample_patches
from lfw import lfwc_paths

__author__ = 'dracz'


epsilon = 0.1  # default regularization value


class PCA:
    """An implementation of principal components analysis"""

    def __init__(self, data):
        """
        Initialize the PCA given the specified training data.

        :param data: Input data to train the model. Data must be 2d ndarray where
        columns represent examples and rows are features of each example
        """
        n, m = data.shape
        self.mean = data.mean(axis=0)
        """mean subtracted from training data"""

        self.x = data - self.mean
        """the zero-mean input data used to fit the model"""

        self.sigma = np.dot(self.x, self.x.T) / m
        """the covariance matrix of the input data"""

        self.u, self.s, self.v = np.linalg.svd(self.sigma, full_matrices=False)
        """the eigenvectors and singular values of the covariance matrix"""

    def reduce(self, x=None, retain=0.9, k=None):
        """
        Return a reduced dimension version of column vector/matrix x and the mean value(s)
        that were subtracted before processing.

        :param x: ndarray of column vector or matrix where columns are examples to reduce
        If not specified, then reduce the data used to train

        :param retain: The amount of variance to retain. Ignored if `k` is not None
        :param k: The number of principal components to retain, or None to specify by `retain`
        :return: A 2-tuple of (reduced_data, mean_subtracted)
        """
        if k is not None and k > len(self.s):
            raise ValueError("k must be <= number of singular values")
        k = k if k is not None else self.find_k(retain)
        if x is None:
            x = self.x
        mean = x.mean(axis=0)
        return np.dot(self.u[:, :k].T, x - mean), mean

    def reconstruct(self, x_tilde, mean=0):
        """
        Reconstruct the data from it's reduced representation x_tilde.
        Optionally add the mean back to the vector before returning.
        Input data is expected in column major order.

        :param x_tilde: The reduced-dimension version of data column vector or matrix
        :param mean: The mean that was subtracted from the original input before reduce
        :return: The reconstructed data vector
        """
        u_reduced = self.u[:, :x_tilde.shape[0]]
        dp = np.dot(u_reduced, x_tilde)
        return dp + mean

    def retaining(self, k):
        """
        Determine how much variance is preserved by the specified value of k
        :param k: The number of principal components to retain
        :return: The amount of variance retained by the specified k value
        """
        return self.s[:k].sum()/self.s.sum()

    def find_k(self, retain=0.99):
        """
        Find smallest k value that preserves the specified amount of variance
        :param retain: How much variance to retain
        :return: k value
        """
        if retain >= 1:
            return len(self.s)
        tot = self.s.sum()
        s_sum = 0.0
        for i in range(len(self.s)):
            s_sum += self.s[i]
            if s_sum / tot >= retain:
                return i+1
        raise ValueError("sum of singular values < {}".format(retain))

    def whiten_pca(self, x=None, eps=epsilon):
        """
        PCA whiten the column vector or matrix x

        :param x: ndarray of data to whiten.
        Either 1-d column vector, or 2-d matrix of column vectors
        If None, then whiten the training data

        :param eps: Regularization parameter
        :return: PCA whitened version of the data and mean value(s) subtracted from input
        """
        if x is None: x = self.x
        d = np.diag(1. / np.sqrt(self.s + eps))
        mean = x.mean(axis=0)
        return np.dot(np.dot(d, self.u.T), x - mean), mean

    def whiten_zca(self, x=None, eps=epsilon):
        """
        ZCA whiten the column vector or matrix x

        :param x: ndarray of data to whiten.
        Either 1-d column vector, or 2-d matrix of column vectors
        If None, then whiten the training data

        :param eps: Regularization parameter
        :return: ZCA whitened version of the data and mean value(s) subtracted from input
        """
        if x is None: x = self.x
        tmp = np.dot(self.u, np.diag(1. / np.sqrt(self.s + eps)))
        tmp = np.dot(tmp, self.u.T)
        mean = x.mean(axis=0)
        return np.dot(tmp, x-mean), mean


def test_pca_input(sh, m):
    """
    Generate input data from labeled faces data
    :param sh: The shape of the image tiles to sample
    :param m: The number of examples to sample
    :return: 2d ndarray where columns are examples
    """
    print("loading {} {} images...".format(m, sh))
    return np.asarray([p.flatten() for p in list(sample_patches(lfwc_paths(), sh, m))]).T


def pca_faces_image(sh=(64, 64), m=500, n_faces=20, k_vals=None,
                    out_dir="./img", test_in_sample=True):
    """
    Visualize the mean face images, then generate an image of faces
    reduced and reconstructed retaining various top-k components
    """
    tot = m if test_in_sample else m + n_faces
    data = test_pca_input(sh, tot)
    x = data[:, :m]

    test_x = data[:, :n_faces] if test_in_sample else data[:, m:]

    pca = PCA(x)

    k_max = pca.find_k(0.99)
    k_min = pca.find_k(0.1)

    if k_vals is None:
        k_vals = [int(k) for k in np.linspace(k_min, k_max, n_faces)]

    img = Image.new("L", (sh[0]*(len(k_vals) + 1), sh[1]*n_faces))

    print("retaining top {}...".format(k_vals))

    # plot and save sample of faces and various levels of reconstruction

    for i in range(test_x.shape[1]):
        img.paste(Image.fromarray(x[:, i].reshape(sh)), (0, i*sh[1]))

        for j, k in enumerate(sorted(k_vals, reverse=True)):
            x_tilde, mean = pca.reduce(test_x[:, i], k=k)
            x_hat = pca.reconstruct(x_tilde, mean=mean).reshape(sh)
            img.paste(Image.fromarray(x_hat), ((j+1)*sh[0], i*sh[1]))

    img.show()

    fn = "pca_faces{}.png".format("_in_sample" if test_in_sample else "_out_sample")
    img.save(os.path.join(out_dir, fn))

    # show the mean face image
    plt.axis('off')
    plt.gray()
    plt.imshow(pca.u[:, 0].reshape(sh))
    plt.savefig(os.path.join(out_dir, "pca_mean_face_{}".format(m)),
                bbox_inches='tight')
    plt.show()


def test_pca(sh=(12, 12), m=10, retain=0.9):
    """
    Show plot demonstrating that the covariance matrix of x_tilde
    is a diagonal matrix with descending values
    :param sh: shape of image patches to test
    :param m: The number of samples to load
    :param retain: Percentage of variance to retain
    """
    x = test_pca_input(sh, m)
    pca = PCA(x)

    xt, mean = pca.reduce(x, retain=retain)
    cov = np.dot(xt, xt.T)

    plt.jet()
    plt.imshow(cov)
    plt.show()


def test_pca_white(sh=(12, 12), m=500, eps=0.1, rc=(10, 10), out_dir="img"):
    # Visualize examples in r x c grid of pca whitened images.
    # Also show that the covariance matrix of whitening data
    # is a diagonal matrix with descending values
    x = test_pca_input(sh, m)
    pca = PCA(x)

    x_white, mean = pca.whiten_pca(x, eps=eps)

    rows, cols = rc

    f, axes = plt.subplots(rows, cols, sharex='col', sharey='row')

    plt.subplots_adjust(hspace=0.1, wspace=0)
    plt.jet()

    for r in range(rows):
        for c in range(cols):
            axes[r][c].imshow(x_white[:, r*cols + c].reshape(sh))
            axes[r][c].axis('off')

    path = os.path.join(out_dir, "pca_filters_{}_{}_{}.png".format(sh, m, eps))
    print("saving {}...".format(path))
    plt.savefig(path, bbox_inches='tight')

    cv = np.dot(x_white, x_white.T)
    plt.clf()
    plt.imshow(cv)
    plt.show()


def test_zca_white(sh=(12, 12), m=500, n_faces=20, out_dir="img", eps=None, show=False):
    """
    Test the ZCA whitening by visualizing at different levels of epsilon
    :param sh: The shape of the tiles
    :param m: The number of examples
    :param n_faces: The number of faces to visualize
    :param out_dir: The directory to write output image
    """

    if eps is None:
        eps = [10**-i for i in np.linspace(-3, 10, 20)]

    x = test_pca_input(sh, m)
    pca = PCA(x)

    f, axes = plt.subplots(n_faces, len(eps)+1, sharex='col', sharey='row')
    plt.subplots_adjust(hspace=0.01, wspace=0.01)

    for r in range(n_faces):
        plt.gray()
        axes[r][0].imshow(x[:, r].reshape(sh))
        axes[r][0].axis('off')

    for c, ep in enumerate(eps):
        print("whitening with eps = {}".format(ep))
        x_white, mean = pca.whiten_zca(x, eps=ep)
        i = 0
        for r in range(n_faces):
            plt.gray()
            axes[r][c+1].imshow(x_white[:, r].reshape(sh))
            axes[r][c+1].axis('off')
            i += 1

    if out_dir is not None:
        fn = "zca_faces_{}_{}_{}-{}.png".format(m, sh, eps[-1], eps[0])
        path = os.path.join(out_dir, fn)
        print("saving {}...".format(path))
        plt.savefig(path, bbox_inches='tight')

    if show:
        plt.show()


def pca_faces():
    m_faces = 5000
    n_faces = 20
    face_shape = (12, 12)

    pca_faces_image(sh=face_shape, m=m_faces, n_faces=n_faces, test_in_sample=True)
    pca_faces_image(sh=face_shape, m=m_faces, n_faces=n_faces, test_in_sample=False)


def test_whitening():
    m_patches = 300000
    patch_sizes = [(i, i) for i in np.arange(8, 17)]

    for sh in patch_sizes:
        test_pca_white(sh=sh, m=m_patches)
        test_zca_white(sh=sh, m=m_patches)


def main():
    pca_faces()
    test_pca()
    test_whitening()
    test_zca_white()


if __name__ == "__main__":
    main()
