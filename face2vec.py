import itertools

from lfw import lfwc_paths, sample_patches
from images import scale_input
from dae import DenoisingAutoencoder
from nn import render_filters

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

    plt.show()


def train_dae(n_samples=100000, patch_shape=(14, 14), n_hidden=64, corruption_rate=0.3,
              learning_rate=0.1, n_epochs=15, batch_size=20):
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

    patches = get_input(tile_shape=patch_shape, n_samples=n_samples)

    n_visible = patch_shape[0] * patch_shape[1]

    da = DenoisingAutoencoder(n_visible, n_hidden)

    da.train(theano.shared(patches),
             corruption_rate=corruption_rate,
             learning_rate=learning_rate,
             n_epochs=n_epochs,
             batch_size=batch_size)

    return da


def main():
    # Train DenoisingAutoencoders using various noise rates.
    # Render sample of input data
    # Render and save image of learned filters

    patch_shape = (12, 12)
    rates = list(np.linspace(0, 0.9, 10))
    for rate in rates:
        da = train_dae(n_samples=100000, corruption_rate= rate,
                       batch_size=100, patch_shape=patch_shape)
        fn = "img/face_patch_filters_{}_{}.png".format(patch_shape, rate)
        w = da.w.get_value(borrow=True)
        render_filters(w, patch_shape, image_file=fn)


if __name__ == "__main__":
    main()
