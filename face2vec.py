import itertools

from lfw import lfwc_paths, sample_patches
from images import scale_input, tile_images_mat
from dae import DenoisingAutoencoder
from nn import render_filters
from transform import PCA
import numpy as np
from matplotlib import pyplot as plt
import theano

__author__ = 'dracz'


def get_input(paths=lfwc_paths(), n_samples=100000, tile_shape=(14, 14)):
    """get 2d ndarray of input vectors in rows"""
    x = [p.flatten() for p in sample_patches(paths, tile_shape, n_samples)]
    a = np.asarray(x, dtype=theano.config.floatX)
    return scale_input(a)


def display_patches(patches, sh, rows=10, cols=10):
    """
    Plot sample of images patches
    :param patches: The patches to display as matrix of columns vectors
    :param sh: The shape of the tiles
    :param rows: The number of rows to display
    :param cols: The number of columns to display
    """
    f, axes = plt.subplots(rows, cols, sharex='col', sharey='row')
    plt.subplots_adjust(hspace=0.01, wspace=0)

    for i, (r, c) in enumerate(itertools.product(range(rows), range(cols))):
        plt.gray()
        axes[r][c].imshow(patches[i, :].reshape(sh))
        axes[r][c].axis('off')
    plt.draw()
    plt.show(block=False)


def train_dae(n_samples=50000, patch_shape=(14, 14), n_hidden=64, corruption_rate=0.3,
              learning_rate=0.1, n_epochs=15, batch_size=200, whiten=0.1,
              show=False, save=None, stop_diff=0.001):
    """
    Train denoising auto-encoder on face samples
    :param n_samples: Number of face sample to use for training
    :param patch_shape: The shape of each input image patch
    :param n_hidden: The number of hidden units
    :param corruption_rate: The amount of noise to introduce [0,1]
    :param learning_rate: The learning rate for gradient descent
    :param n_epochs: The number of training epochs
    :param batch_size: The mini-batch size
    :return: A trained DenoisingAutoencoder
    """

    print("Sampling {} input patches of shape {}...".format(n_samples, patch_shape))
    patches = get_input(tile_shape=patch_shape, n_samples=n_samples)

    display_patches(patches, patch_shape, rows=20, cols=20)

    if whiten:
        print("Whitening with epsilon = {}".format(whiten))
        pca = PCA(patches.T)

        x, mean = pca.whiten_zca(eps=whiten)
        x = scale_input(x.T)
        patches = x

        display_patches(patches, patch_shape, rows=20, cols=20)

    n_visible = patch_shape[0] * patch_shape[1]

    da = DenoisingAutoencoder(n_visible, n_hidden)

    da.train(theano.shared(patches),
             corruption_rate=corruption_rate,
             learning_rate=learning_rate,
             n_epochs=n_epochs,
             batch_size=batch_size,
             stop_diff=stop_diff)

    if show or save:
        show_save(da.w.get_value(borrow=True), save=save, show=show, whiten=whiten,
                  n_hidden=n_hidden, patch_shape=patch_shape, corruption_rate=corruption_rate)

    return da


def show_save(w, show=False, save=False, whiten=None, n_hidden=None, patch_shape=None, corruption_rate=None):
    """show/save the learned weights"""
    if save:
        wn = "" if not whiten else "_white_{}".format(whiten)
        image_file = "img/face_patch_filters_{}_{}_{}{}.png".format(patch_shape, n_hidden, corruption_rate, wn)
    else:
        image_file = None
    render_filters(w, patch_shape, image_file=image_file, show=show)


def sweep_params(epsilons=None, rates=None, tile_range=None,
                 n_hidden_range=None, save=True, show=False):
    """
    Sweep parameters for epsilon, tile_shape, n_hidden, corruption_rates
    and save the learned filters
    """
    if epsilons is None:
        epsilons = [1, 0.5, 0.1, 0.05, 0]

    if rates is None:
        rates = [0.3]

    if tile_range is None:
        tile_range = range(8, 11)

    for eps in epsilons:
        for rate in rates:
            for w in tile_range:
                if n_hidden_range is None:
                    n_hidden_range = [int(i**2) for i in np.linspace(4, w, 4)]

                for h in n_hidden_range:
                    train_dae(n_hidden=h, patch_shape=(w, w), whiten=eps,
                              corruption_rate=rate, save=save, show=show)


if __name__ == "__main__":
    sweep_params(epsilons=[1, 0.5, 0.1, 0], tile_range=[20],
                 n_hidden_range=[36, 49, 100, 144],
                 show=True, save=True)
