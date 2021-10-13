import cv2
import numpy as np
from deep_privacy import logger
from deep_privacy.inference.deep_privacy_anonymizer import DeepPrivacyAnonymizer
from deep_privacy.build import build_anonymizer, available_models


def lsb_encoding(image, x0, y0, x1, y1, message):
    encoded = np.copy(image)
    idx = len(message)-1
    curr_bit = -1

    for x in range(x0, x1):
        for y in range(y0, y1):
            for c in range(3):
                #curr_bit = message >> idx & 1
                if idx >= 0:
                    curr_bit = message[idx]
                    idx -= 1
                    # replaces lsb with curr_bit
                    encoded[y, x, c] = image[y, x, c] & ~1 | curr_bit
                else:
                    encoded[y, x, c] = image[y, x, c] & ~1 | 0

    return encoded


def lsb_decode(encoded, x0, y0, x1, y1):
    #message = 0
    message = ""
    for x in reversed(range(x0, x1)):
        for y in reversed(range(y0, y1)):
            for c in reversed(range(3)):
                # message = (message << 1) + (encoded[y, x, c] & 1) #add lsb to message
                message += str(encoded[y, x, c] & 1)

    return message[message.find("1"):]


def get_face_boundaries(model, image):
    anonymizer, cfg = build_anonymizer(
        model, opts=None, config_path=None,
        return_cfg=True)
    annotations = anonymizer.get_detections([image])
    x0, y0, x1, y1 = annotations[0].bbox_XYXY[0]
    #cv2.imwrite(targetCutPath, image[y0:y1, x0:x1][:, :, ::-1])
    return x0, y0, x1, y1


def add_face_to_image(face, image, x0, y0, x1, y1):
    new_image = np.copy(image)
    for x in range(x0, x1):
        for y in range(y0, y1):
            for c in range(3):
                new_image[y, x, c] = face[y - y0, x - x0, c]
    return new_image
