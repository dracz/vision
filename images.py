__author__ = 'dracz'

import math
import random
import cv2
import numpy as np
import theano

from PIL import Image


""" Various image manipulation utilities """


def load_images(paths):
    """load images from paths into list of 2d ndarray"""
    return [cv2.imread(path, 0) for path in paths]


def load_matrix_2d(paths):
    """ load images from paths into columns in n x m ndarray """
    return np.asarray([img.flatten() for img in load_images(paths)]).T


def load_matrix_3d(paths):
    """
    Load images from paths into m x w x h 3d ndarray
    """
    return np.asarray(load_images(paths))


def sample_patches(paths, patch_size, n_patches):
    """
    Sample random gray-scale patches from face images found in paths list
    :param patch_size: the (w, h) of the image patch to sample
    :return: iterator over 2d ndarray of grayscale values
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


def tile_images(img_tiles, tile_shape, tiling_shape=None):
    """
    Generate an image composed of the img_tiles
    :param img_tiles: list of 2-d ndarray of grey-scale image tiles
    :param tile_shape: The shape of the img_tiles (n_pixels_w, n_pixels_h)
    :param tiling_shape: The shape of the tiling (n_tiles_w, n_tiles_h)
    :return: an Image
    """
    if not tiling_shape:
        w = int(math.floor(math.sqrt(len(img_tiles))))
        h = w
    else:
        w, h = tiling_shape

    tile_w, tile_h = tile_shape  # shape of image tile
    shape = (w * tile_w, h * tile_h)
    img = Image.new("L", shape, "white")

    for i, mat in enumerate(img_tiles):
        tile = Image.fromarray(mat)
        if i == w*h:
            break
        r, c = i/w, i % w
        box = (c*tile_w, r*tile_h, c*tile_w + tile_w, r*tile_h + tile_h)
        img.paste(tile, box)
    return img


