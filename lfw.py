
import glob
import os
import re
import string
import sys
import random
import timeit

import cv2
import numpy as np

import detectors
from images import load_matrix_2d, all_patches
from images import sample_patches, tile_images

__author__ = 'dracz'

LFW_ROOT = "../../data/lfw"
LFWC_ROOT = "../../data/lfwcrop_grey"  # contains 13,232 images


def load_lfw(path=LFW_ROOT, shuffle=True, limit=-1, scale=False):
    return load_matrix_2d(lfw_paths(path, shuffle=shuffle, limit=limit))


def load_lfwc(path=LFWC_ROOT, shuffle=True, limit=-1, scale=False):
    return load_matrix_2d(lfwc_paths(path, shuffle=shuffle, limit=limit))


def load_lfwc_patches(patch_shape, shuffle=True, cache=True, cache_dir="./cache"):
    t1 = timeit.default_timer()

    cache_file = os.path.join(cache_dir, "lfwc_patches_{}.npy".format(patch_shape))

    if cache and os.path.exists(cache_file):
        print("Loading cached patches from {}...".format(cache_file))
        patches = np.load(cache_file)
        t2 = timeit.default_timer()
        print("Loaded {:,} patches from {} in {:.2f}".format(len(patches), cache_file, t2-t1))
        return patches

    print("Loading all lfwc patches of shape {}...".format(patch_shape))
    patches = np.asarray(list(all_patches(lfwc_paths(), patch_shape, shuffle=shuffle)))

    t2 = timeit.default_timer()
    print("Loaded {:,} patches in {:.2f}".format(len(patches), t2-t1))

    if cache:
        print("Saving patches to {}...".format(cache_file))
        np.save(cache_file, patches)

    return patches


def lfw_paths(path=LFW_ROOT, shuffle=True, limit=-1):
    """
    Get "labeled faces in the wild" (lfw) image files downloaded from:
    http://vis-www.cs.umass.edu/lfw/lfw.tgz
    :param path: local path to the root of the downloaded images
    :param shuffle: whether to read in random order; default=True
    :return: iterator of path-name strings
    """
    paths = glob.glob(os.path.join(path, "*/*.jpg"))
    if shuffle:
        random.shuffle(paths)
    return paths[:limit]


def lfwc_paths(path=LFWC_ROOT, shuffle=True, limit=-1):
    """
    Get "cropped labeled faces in the wild" (lfwc) image files downloaded from:
    http://conradsanderson.id.au/lfwcrop/lfwcrop_grey.zip
    :param path: local path to the root of the downloaded images
    :param shuffle: whether to read in random order; default=True
    :return: iterator of path-name strings
    """
    paths = glob.glob(os.path.join(path, "faces/*.pgm"))

    if shuffle:
        random.shuffle(paths)
    return paths[:limit]


def img_info(path):
    file_name = path[string.rfind(path, "/") + 1:]
    match = re.match(r'^(.*)_(\d+).\w+$', file_name)
    name, number = match.group(1), match.group(2)
    return path, name, number


def show_images(paths, detector=None):
    """
    Show lfw images from path one at a time,
    run through detectors, wait for keyboard input
    """
    print "Press 'q' to exit, 'b' to view previous, or any other key to advance"
    i = 0

    while True:
        path, name, number = img_info(paths[i])
        img = cv2.imread(path)
        show_image("{} {}".format(name, number), img, detector)

        k = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if k == ord('q'):
            sys.exit(0)
        elif k == ord('b'):
            if i > 0:
                i -= 1
            else:
                print("Reached the first image")
        elif i < len(paths) - 1:
            i += 1
        else:
            print("Reached the last image")


def show_image(title, img, detector=None, size=None):
    """
    :param title: to use for title of image
    :param img: ndarray of image
    :param detector: detector to run image through
    """
    if detector is None:
        display_img = img
    else:
        display_img = img.copy()
        detected = detector.detect(img)
        detectors.draw_detected(display_img, detected)
    if size is not None:
        display_img = cv2.resize(display_img, size)
    cv2.imshow(title, display_img)


if __name__ == "__main__":
    args = sys.argv[1:]
    cmd = args[0] if len(args) > 0 else "patches"

    if cmd == "show":
        show_images(lfwc_paths())

    elif cmd == "sample":
        n_samples = 2500
        tile_shape = (20, 20)
        patches = list(sample_patches(lfwc_paths(), tile_shape, n_samples))
        tiled = tile_images(patches, tile_shape, show=True)

    elif cmd == "patches":
        shapes = [(i, i) for i in range(8, 12)]
        for sh in shapes:
            patches = load_lfwc_patches(sh, cache_dir="./cache")
            img = tile_images(patches, sh, output_shape=(100, 100), show=True)
            fn = "./img/lfwc_patches_{}.png".format(sh)
            print("saving {}...".format(fn))
            img.save(fn)
