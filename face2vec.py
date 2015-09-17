
import glob, os, random

import cv2, theano
import numpy as np
from nn import render_filters
from images import plot_images, sample_patches
from lfw import LFWC_PATTERN

from dae import DenoisingAutoencoder


def get_input(input_pattern=LFWC_PATTERN,
              n_samples=5000, patch_shape=(12, 12),
              n_images=100, epsilon=0.1, norm_axis=0):
    """
    Get input data from lfwc patches
    :param input_pattern: The pattern to glob for image files
    :param n_images: The number of images to draw samples from
    :param n_samples: The number of samples to extract
    :param patch_shape: The shape of the patches to sample
    :param epsilon: Regularization for whitening, or None to disable whitening
    :param norm_axis: Whether to mean normalize across each feature (0) or each patch (1)
    :return: m x n ndarray of flattened images
    """

    paths = glob.glob(input_pattern)
    random.shuffle(paths)
    paths = paths[:n_images]

    print("Sampling {} patches from {} images...".format(n_samples, len(paths)))

    # read images into a m x img_row x img_col ndarray
    imgs = np.asarray([cv2.imread(path, 0) for path in paths], dtype=theano.config.floatX)
    imgs /= 255.  # scale to [0,1] # why is this so crucial?

    # sample patches into a 3d array
    patches = sample_patches(imgs, patch_shape=patch_shape, n_samples=n_samples)

    # flatten each image to 1d
    patches = patches.reshape(patches.shape[0], patch_shape[0]*patch_shape[1])

    if norm_axis == 0:
        # subtract per feature (pixel) means across the data set (m x n)
        normed = patches - patches.mean(axis=0)
    else:
        # remove mean val of each patch
        normed = patches - patches.mean(axis=1)[:, np.newaxis]

    if epsilon is None:
        return normed

    return whiten(normed, epsilon=epsilon)


def whiten(data, epsilon=0.1):
    """whiten the m x n array of images"""
    U, S, V = np.linalg.svd(np.cov(data.T), full_matrices=False)
    tmp = np.dot(U, np.diag(1. / np.sqrt(S + epsilon)))
    tmp = np.dot(tmp, U.T)
    z = np.dot(tmp, data.T)
    return z.T


def train_dae(patch_shape=(20,20),
              n_hidden=400,
              n_samples=50000,
              batch_size=500,
              epsilon=0.1,
              show_input=False,
              show_filters=True,
              img_dir=None,
              norm_axis=0):

    """ train a denoising autoencoder """

    data = get_input(n_samples=n_samples, patch_shape=patch_shape, epsilon=epsilon, norm_axis=norm_axis)
    if show_input:
        plot_images(data, shape=patch_shape)

    da = DenoisingAutoencoder(n_visible=patch_shape[0]*patch_shape[1],
                              n_hidden=n_hidden)

    da.train(theano.shared(data), batch_size=batch_size)

    if img_dir is not None:
        fn = os.path.join(img_dir, "face_filters_{}_{}_{}_{}_{}_{}.png"
                          .format(patch_shape, n_hidden, n_samples, batch_size, epsilon, norm_axis))
    else:
        fn = None

    render_filters(da.w.get_value(borrow=True), patch_shape, show=show_filters, image_file=fn)


def sweep(patch_shapes=[(24, 24)],
          n_samples=20000,
          batch_size=20,
          epsilons=[0.1, 0.01],
          n_hiddens=[100, 400],
          norm_axes=[0],
          img_dir="./img/face_filters"):

    """train autoencoders using range of params, render and save filters"""

    for patch_shape in patch_shapes:
        for eps in epsilons:
            for axis in norm_axes:
                for n_hidden in n_hiddens:
                    train_dae(patch_shape=patch_shape,
                              batch_size=batch_size,
                              n_samples=n_samples,
                              n_hidden=n_hidden,
                              epsilon=eps,
                              img_dir=img_dir,
                              norm_axis=axis)


if __name__ == "__main__":
    sweep()
