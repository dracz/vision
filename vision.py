
import glob, itertools, os, random

import cv2, theano
import numpy as np
from matplotlib import pyplot as plt

from sae import SparseAutoencoder
from dae import DenoisingAutoencoder

LFWC_ROOT = '../../data/lfwcrop_grey'  # contains 13,232 images


def plot_images(imgs, rows=10, cols=10, block=True, shape=None, fname=None, show=True):
    """
    Plot a sample of the images
    :param imgs: 3d ndarray m x r x c of images
    :param rows: The number of rows to display
    :param cols: The number of columns to display
    :param shape: If shape, then expect imgs is 2d array and reshape
    """
    if fname is None and show is False:
        return

    f, axes = plt.subplots(rows, cols, sharex='col', sharey='row')
    plt.subplots_adjust(hspace=0.01, wspace=0)

    for i, (r, c) in enumerate(itertools.product(range(rows), range(cols))):
        plt.gray()
        if shape is not None:
            img = imgs[i,:].reshape(shape)
        else:
            img = imgs[i,:,:]
        axes[r][c].imshow(img)
        axes[r][c].axis('off')
    plt.draw()
    if fname:
        plt.savefig(fname)
    if show:
        plt.show(block=block)


def sample_patches(imgs, n_samples, patch_shape):
    """
    Sample n_samples of patch_shape from imgs (with replacement)
    :param imgs: ndarray of shape [n_imgs, n_img_rows, n_img_cols]
    :param patch_shape: The shape of the patches to sample
    :return: ndarray of shape [n_samples, patch_shape[0], patch_shape[1]]
    """

    n_images, n_img_rows, n_img_cols = imgs.shape
    n_patch_rows, n_patch_cols = patch_shape

    assert (n_patch_rows <= n_img_rows)
    assert (n_patch_cols <= n_img_cols)

    patches = np.zeros((n_samples, n_patch_rows, n_patch_cols))

    for n in range(n_samples):
        i = np.random.randint(0, n_images)
        r = np.random.randint(0, n_img_rows - n_patch_rows)
        c = np.random.randint(0, n_img_cols - n_patch_cols)
        patches[n,:,:] = imgs[i, r:r + n_patch_rows, c:c + n_patch_cols]

    return patches


def get_input(input_root=LFWC_ROOT,
              pattern='faces/*.pgm',
              n_samples=5000,
              patch_shape=(12, 12),
              n_images=100,
              epsilon=0.1, norm_axis=0):

    """get input data """

    input_pattern = os.path.join(input_root, pattern)
    paths = glob.glob(input_pattern)
    random.shuffle(paths)
    paths = paths[:n_images]

    print("Sampling {} patches from {} images...".format(n_samples, len(paths)))

    # read images into a m x img_row x img_col ndarray
    imgs = np.asarray([cv2.imread(path, 0) for path in paths], dtype=theano.config.floatX)
    imgs /= 255.  # scale to [0,1] # why is this so crucial?

    # sample patches into a 3d array
    patches = sample_patches(imgs, n_samples, patch_shape)

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


from nn import render_filters


def train(patch_shape=(20,20),
          n_hidden=400,
          n_samples=50000,
          batch_size=500,
          epsilon=0.1,
          show_input=False,
          show_filters=True,
          img_dir=None):

    data = get_input(n_samples=n_samples, patch_shape=patch_shape, epsilon=epsilon)
    if show_input:
        plot_images(data, shape=patch_shape)

    da = DenoisingAutoencoder(n_visible=patch_shape[0]*patch_shape[1],
                              n_hidden=n_hidden)
    da.train(theano.shared(data), batch_size=batch_size)

    if img_dir is not None:
        fn = os.path.join(img_dir, "face_filters_{}_{}_{}_{}_{}.png"
                          .format(patch_shape, n_hidden, n_samples, batch_size, epsilon))
    else:
        fn = None

    render_filters(da.w.get_value(borrow=True), patch_shape, show=show_filters, image_file=fn)


def sweep(patch_shapes=[(20, 20), (12, 12), (8, 8)],
          n_samples=20000,
          batch_size=20,
          epsilons=[0, 0.1, 0.01],
          img_dir="./img/face_filters_3"):

    for patch_shape in patch_shapes:
        for eps in epsilons:
            train(patch_shape=patch_shape, epsilon=eps, batch_size=batch_size, img_dir=img_dir, n_samples=n_samples)

if __name__ == "__main__":
    sweep()



