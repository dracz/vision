__author__ = 'dracz'

import cv2

import detectors


def loop():
    video_capture = cv2.VideoCapture(0)
    detector = detectors.Detector()
    while True:
        ret, img = video_capture.read()

        display_img = img.copy()
        detected = detector.detect(img)
        detectors.draw_detected(display_img, detected)

        cv2.imshow("Video", display_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    loop()

