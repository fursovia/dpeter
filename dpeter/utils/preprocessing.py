import cv2
import numpy as np


def normalization(imgs):
    """Normalize list of images"""

    imgs = np.asarray(imgs).astype(np.float32)
    imgs = np.expand_dims(imgs / 255, axis=-1)
    return imgs


def rotate_maybe(img: np.ndarray) -> np.ndarray:
    w, h = img.shape
    if w > h * 2:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return img
