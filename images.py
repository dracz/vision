
import math
import random
import cv2
import numpy as np
import itertools

from PIL import Image

__author__ = 'dracz'

""" Various image manipulation utilities """


def load_images(paths):
    """ Load images from paths into list of 2d ndarray """
    return [cv2.imread(path, 0) for path in paths]


def load_matrix_2d(paths, scale=False):
    """ Load images from paths into columns in n x m ndarray """
    a = np.asarray([img.flatten() for img in load_images(paths)]).T
    if not scale:
        return a
    return scale_input(a)


def load_matrix_3d(paths):
    """ Load images from paths into m x w x h 3d ndarray """
    return np.asarray(load_images(paths))


def sample_patches(paths, patch_size, n_patches):
    """
    Sample random gray-scale patches from face images found in paths list
    :param paths: Paths of input image files
    :param patch_size: the (w, h) of the image patch to sample
    :param n_patches: The total number of patches to load
    :return: Generator producing 2d ndarray of grayscale values
    """
    imgs = load_images(paths)

    for i in range(n_patches):
        # choose random img
        ri = random.randrange(len(imgs))
        img = imgs[ri]

        patch_width, patch_height = patch_size

        # choose random patch
        x = random.randrange(img.shape[0]-patch_width+1)
        y = random.randrange(img.shape[1]-patch_height+1)
        patch = img[x:x+patch_width, y:y+patch_height]

        yield patch


def all_patches(paths, patch_shape, shuffle=True):
    """
    Load all patches of the specified shape from the images found in paths
    :return: A generator of 2d ndarray of patch_shape
    """
    mat = load_matrix_3d(paths)
    img_rows, img_cols = mat.shape[1:]
    patch_rows, patch_cols = patch_shape
    row_ranges = [(r, r+patch_rows) for r in range(img_rows-patch_rows)]
    col_ranges = [(c, c+patch_cols) for c in range(img_cols-patch_cols)]
    img_inds = range(mat.shape[0])

    inds = list(itertools.product(img_inds, row_ranges, col_ranges))
    if shuffle:
        random.shuffle(inds)

    for i, r_rng, c_rng in inds:
        r_start, r_end = r_rng
        c_start, c_end = c_rng
        yield mat[i, r_start:r_end, c_start:c_end]


def tile_images(img_tiles, patch_shape, output_shape=None, show=False):
    """
    Generate an image composed of the img_tiles
    :param img_tiles: list of 2-d ndarray of grey-scale image tiles
    :param patch_shape: The shape of the img_tiles (n_pixels_w, n_pixels_h)
    :param output_shape: The shape of the tiling (n_tiles_w, n_tiles_h)
    :param show: Whether to display the image
    :return: An instance of PIL Image
    """
    if not output_shape:
        w = int(math.floor(math.sqrt(len(img_tiles))))
        h = w
    else:
        w, h = output_shape

    tile_w, tile_h = patch_shape  # shape of image tile
    shape = (w * tile_w, h * tile_h)
    img = Image.new("L", shape, "white")

    for i in range(w*h):
        tile = Image.fromarray(img_tiles[i])
        if i == w*h:
            break
        r, c = i/w, i % w
        box = (c*tile_w, r*tile_h, c*tile_w + tile_w, r*tile_h + tile_h)
        img.paste(tile, box)

    if show:
        img.show()

    return img

