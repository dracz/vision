
import itertools
import math

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

__author__ = 'dracz'

"""
Various image manipulation utilities
"""


def load_images(paths):
    """ Load images from paths into list of 2d ndarray """
    return [cv2.imread(path, 0) for path in paths]


def load_matrix_2d(paths, scale=False):
    """ Load images from paths into columns in n x m ndarray """
    a = np.asarray([img.flatten() for img in load_images(paths)]).T
    return a


def load_matrix_3d(paths):
    """ Load images from paths into m x w x h 3d ndarray """
    return np.asarray(load_images(paths))


def sample_patches(imgs, patch_shape=(12, 12), n_samples=10000):
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


def tile_images(imgs, patch_shape=None, output_shape=None, show=False):
    """
    Generate an image composed of the img_tiles
    :param imgs: m x r x c ndarray of image patches
    :param output_shape: The shape of the tiling (n_tiles_w, n_tiles_h)
    :param show: Whether to display the image
    :return: An instance of PIL Image
    """

    m, n_patch_rows, n_patch_cols = imgs.shape

    if not output_shape:
        w = int(math.floor(math.sqrt(m)))
        h = w
    else:
        w, h = output_shape

    tile_w, tile_h = patch_shape  # shape of image tile
    img = Image.new("L", (w * tile_w, h * tile_h), "white")

    for i in range(w*h):
        if i >= m: break
        tile = Image.fromarray(imgs[i,:,:])
        if i == w*h:
            break
        r, c = i/w, i % w
        box = (c*tile_w, r*tile_h, c*tile_w + tile_w, r*tile_h + tile_h)
        img.paste(tile, box)

    if show:
        img.show()

    return img


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
