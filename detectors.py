__author__ = 'dracz'

from os import path
from collections import OrderedDict

import numpy as np
import cv2

default_params = dict(scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

# default_classes = ["faces", "eyes", "smiles"]
default_classes = ["faces"]


class Detector:
    def __init__(self, kinds=default_classes):
        self.detectors = OrderedDict()

        if "faces" in kinds:
            face_cascade_file = "/opt/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
            assert path.exists(face_cascade_file)
            self.detectors["faces"] = (cv2.CascadeClassifier(face_cascade_file), default_params)

        if "eyes" in kinds:
            assert "faces" in kinds
            eye_cascade_file = "/opt/local/share/OpenCV/haarcascades/haarcascade_eye.xml"
            assert path.exists(eye_cascade_file)
            self.detectors["eyes"] = (cv2.CascadeClassifier(eye_cascade_file), default_params)

        if "smiles" in kinds:
            assert "faces" in kinds
            smile_cascade_file = "/opt/local/share/OpenCV/haarcascades/haarcascade_smile.xml"
            assert path.exists(smile_cascade_file)
            self.detectors["smiles"] = (cv2.CascadeClassifier(smile_cascade_file), default_params)

    def detect(self, img):
        """
        :param img: input image as numpy.ndarray
        :return: dict mapping results to detector type
        """
        ret = dict()
        keys = self.detectors.keys()

        if len(keys) == 0:
            return ret

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if "faces" in keys:
            ret["faces"] = self.detect_class(gray, "faces")

        if "eyes" in keys:
            ret["eyes"] = self.detect_within(gray, ret["faces"], "eyes")

        if "smiles" in keys:
            ret["smiles"] = self.detect_within(gray, ret["faces"], "smiles")

        return ret

    def detect_class(self, img, kind):
        cascade, params = self.detectors[kind]
        detected = cascade.detectMultiScale(img, **params)
        return detected

    def detect_within(self, img, rois, kind):
        cascade, params = self.detectors[kind]
        for x, y, w, h in rois:
            detected = cascade.detectMultiScale(img[y:y+h, x:x+w], **params)
            if len(detected) > 0:
                detected[:, 0] += x
                detected[:, 1] += y
            for rect in detected:
                yield (rect)


from collections import defaultdict

# colors for drawing boxes
colors = defaultdict(lambda: (0, 0, 0),
                     faces=(0, 0, 255),
                     eyes=(255, 0, 0),
                     smiles=(0, 255, 0))


def draw_detected(img, detected):
    """ draw boxes around detected things """
    for key, boxes in detected.items():
        draw_boxes(img, boxes, colors[key])


def draw_boxes(img, boxes, rgb):
    """ draw a box the specified rgb on the img """
    for x,y,w,h in boxes:
        cv2.rectangle(img, (x,y), (x+w, y+h), rgb, 1)