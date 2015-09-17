
import glob
import os
import re
import string
import sys
import random

import cv2

import detectors
from images import load_matrix_3d
from images import sample_patches, tile_images

__author__ = 'dracz'

LFW_PATTERN = "../../data/lfw/*/*.jpg"
LFWC_PATTERN = "../../data/lfwcrop_grey/*/*.pgm"


def load_lfw(pattern=LFW_PATTERN, shuffle=True, limit=-1):
    """load lfw data into 3d ndarray of shape [m, patch_rows, patch_cols]"""
    print("Loading lfw data from {}...".format(pattern))
    return load_matrix_3d(lfw_paths(pattern, shuffle=shuffle, limit=limit))


def load_lfwc(pattern=LFWC_PATTERN, shuffle=True, limit=-1):
    """load lfwc data into 3d ndarray of shape [m, patch_rows, patch_cols]"""
    print("Loading lfwc data from {}...".format(pattern))
    return load_matrix_3d(lfwc_paths(pattern, shuffle=shuffle, limit=limit))


def lfw_paths(pattern=LFW_PATTERN, shuffle=True, limit=-1):
    """
    Get "labeled faces in the wild" (lfw) image files downloaded from:
    http://vis-www.cs.umass.edu/lfw/lfw.tgz
    :param path: local path to the root of the downloaded images
    :param shuffle: whether to read in random order; default=True
    :return: iterator of path-name strings
    """
    paths = glob.glob(pattern)
    if shuffle:
        random.shuffle(paths)
    return paths[:limit]


def lfwc_paths(patter=LFWC_PATTERN, shuffle=True, limit=-1):
    """
    Get "cropped labeled faces in the wild" (lfwc) image files downloaded from:
    http://conradsanderson.id.au/lfwcrop/lfwcrop_grey.zip
    :param path: local path to the root of the downloaded images
    :param shuffle: whether to read in random order; default=True
    :return: iterator of path-name strings
    """
    paths = glob.glob(LFWC_PATTERN)
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


def main(cmd, args):
    """process the command"""
    data_set = "lfw" if len(args) > 0 and args[0] == "lfw" else "lfwc"

    if cmd == "show":
        if len(args) > 0 and args[0] == "lfw":
            show_images(lfw_paths())
        else:
            show_images(lfwc_paths())
        return

    n_samples = 10000
    sh = (20, 20)
    out_shape = (100, 100)
    data = load_lfw() if data_set == "lfw" else load_lfwc()
    out_dir = "./img/"

    if cmd == "sample":
        patches = sample_patches(data, patch_shape=sh, n_samples=n_samples)
        tile_images(patches, patch_shape=sh, output_shape=out_shape, show=True)

    elif cmd == "sweep":
        shapes = [(i, i) for i in range(14, 20, 2)]
        for sh in shapes:
            patches = sample_patches(data, n_samples=n_samples, patch_shape=sh)
            img = tile_images(patches, sh, output_shape=out_shape, show=True)
            fn = os.path.join(out_dir, "{}_patches_{}.png".format(data_set, sh))
            print("saving {}...".format(fn))
            img.save(fn)


if __name__ == "__main__":
    argv = sys.argv[1:]
    cmd = argv[0] if len(argv) > 0 else "show"
    args = [] if len(argv) < 2 else argv[1:]
    main(cmd, args)
