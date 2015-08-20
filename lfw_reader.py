__author__ = 'dracz'

import glob
import os
import re
import string
import sys
import random

import cv2

import detectors

DEFAULT_PATH = "../../data/lfw"


def get_image_files(path, shuffle=True):
    """
    Get "labeled faces in the wild" (lfw) image files downloaded from:
    http://vis-www.cs.umass.edu/lfw/lfw.tgz
    :param path: local path to the root of the downloaded images
    :param shuffle: whether to read in random order; default=True
    :return: iterator of (full_path, person_name, img_number)
    """
    paths = glob.glob(os.path.join(path, "*/*.jpg"))
    if shuffle:
        random.shuffle(paths)

    for file_path in paths:
        file_name = file_path[string.rfind(file_path, "/")+1:]
        match = re.match(r'^(.*)_(\d+).jpg$', file_name)
        name, number = match.group(1), match.group(2)
        yield file_path, name, number


def show_images(path, shuffle=True, detector=None):
    print "Showing images from {}".format(path)
    print "Press 'q' to exit, 'b' to view previous, or any other key to advance"

    img_info = list(get_image_files(path, shuffle))
    ind = 0

    while True:
        file_path, name, number = img_info[ind]
        img = cv2.imread(file_path)

        show_image(name, number, img, detector)

        k = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if k == ord('q'):
            sys.exit(0)
        elif k == ord('b') or k == 63234 and ind > 0:
            ind -= 1
        elif ind < len(img_info) - 1:
            ind += 1
        else:
            print("Reached the last image")


def show_image(name, number, img, detector=None):
    """
    :param name: name of the person in the image
    :param number: the image number
    :param img: ndarray of image
    :param detector: detector to run image through
    """
    scale = 1.5
    if detector is None:
        display_img = img
    else:
        display_img = img.copy()
        detected = detector.detect(img)
        detectors.draw_detected(display_img, detected)

        display_img = cv2.resize(display_img, None, fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)

    cv2.imshow("{} {}".format(name, number), display_img)


if __name__ == "__main__":
    args = sys.argv[1:]
    img_path = args[0] if len(args) > 0 else DEFAULT_PATH
    show_images(img_path, True, detectors.Detector())


