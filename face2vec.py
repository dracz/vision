
import glob, os, random

import cv2, theano
import numpy as np

from nn import render_filters, reconstruct
from images import plot_images, sample_patches, load_matrix_2d
from faces import lfwc_pattern, lfwc_count

import dae, sae


def get_input(input_pattern=lfwc_pattern,
              n_samples=5000, patch_shape=(12, 12),
              n_images=-1, epsilon=0.1, norm_axis=0):
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

    print("Sampling {} {}x{} patches from {} images..."
          .format(n_samples, patch_shape[0], patch_shape[1], len(paths)))

    # read images into a m x img_row x img_col ndarray
    imgs = np.asarray([cv2.imread(path, 0) for path in paths], dtype=theano.config.floatX)
    imgs /= 255.  # scale to [0,1]

    if input_pattern == lfwc_pattern and patch_shape == (64,64) and n_samples == lfwc_count:
        patches = load_matrix_2d(paths)
        epsilon = 0 # disable whitening... too slow

    else:
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

    return whiten_zca(normed, epsilon=epsilon)


def whiten_zca(data, epsilon=0.1):
    """
    zca whiten the m x n data array of images
    :param epsilon: the regularization to use, or 0 to disable whitening
    """
    if not epsilon:
        return data.T
    U, S, V = np.linalg.svd(np.cov(data.T), full_matrices=False)
    tmp = np.dot(U, np.diag(1. / np.sqrt(S + epsilon)))
    tmp = np.dot(tmp, U.T)
    z = np.dot(tmp, data.T)
    return z.T


def train_ae(ae, patch_shape=(20, 20), n_hidden=400, n_samples=50000,
             batch_size=500, epsilon=0.1, show_input=False, show_filters=False,
             sparsity=0.01, beta=0.0, weight_decay=0.0, corruption_rate=0.3,
             img_dir=None, norm_axis=0, n_epochs=5, n_reconstruct=10):

    """ train an autoencoder """
    img_data = get_input(n_samples=n_samples, patch_shape=patch_shape, epsilon=epsilon, norm_axis=norm_axis)
    data = theano.shared(img_data, borrow=True)

    if show_input:
        plot_images(img_data, shape=patch_shape)

    n_vis = patch_shape[0]*patch_shape[1]
    trained = ae.train(data, n_visible=n_vis, n_hidden=n_hidden, batch_size=batch_size, sparsity=sparsity,
                       weight_decay=weight_decay, beta=beta, corruption_rate=corruption_rate, n_epochs=n_epochs)

    if img_dir is not None:
        if ae.__name__ == "dae":
            opts = "_{}".format(corruption_rate)
        elif ae.__name__ == "sae":
            opts = "_{}_{}_{}".format(weight_decay, beta, sparsity)

        fp = os.path.join(img_dir, "ff_{}_{}_{}_{}_{}_{}_{}{}.png"
                          .format(ae.__name__, patch_shape, n_hidden,
                                  n_samples, batch_size, epsilon, norm_axis, opts))
    else:
        fp = None

    render_filters(trained.w1.get_value(borrow=True), patch_shape, show=show_filters, image_file=fp)
    reconstruct(trained, img_data, patch_shape, n_reconstruct)

    return trained


def sweep(autoencoders,
          patch_shapes=[(64, 64)],
          n_samples=lfwc_count,
          batch_size=5,
          epsilons=[0.05],
          n_hiddens=[1000],
          corruption_rates=[0.3],
          betas=[3],
          sparsities=[0.05],
          weight_decays=[0.0001],
          norm_axes=[0],
          n_epochs=10,
          img_dir="./img/face_filters",
          n_reconstruct=40):

    """train autoencoders for specified ranges of parameters,
     render and save filter images"""
    for ae in autoencoders:
        for patch_shape in patch_shapes:
            for eps in epsilons:
                for axis in norm_axes:
                    for n_hidden in n_hiddens:
                        for corruption_rate in corruption_rates:
                            for beta in betas:
                                for sparsity in sparsities:
                                    for weight_decay in weight_decays:
                                        t = train_ae(ae,
                                                     patch_shape=patch_shape,
                                                     batch_size=batch_size,
                                                     n_samples=n_samples,
                                                     n_hidden=n_hidden,
                                                     epsilon=eps,
                                                     corruption_rate=corruption_rate,
                                                     beta=beta,
                                                     sparsity=sparsity,
                                                     weight_decay=weight_decay,
                                                     img_dir=img_dir,
                                                     norm_axis=axis,
                                                     n_epochs=n_epochs,
                                                     n_reconstruct=n_reconstruct)


if __name__ == "__main__":
    sweep([dae])





