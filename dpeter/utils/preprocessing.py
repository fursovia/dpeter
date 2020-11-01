"""
Data preproc functions:
    adjust_to_see: adjust image to better visualize (rotate and transpose)
    augmentation: apply variations to a list of images
    normalization: apply normalization and variations on images (if required)
    preprocess: main function for preprocess.
        Make the image:
            illumination_compensation: apply illumination regularitation
            remove_cursive_style: remove cursive style from image (if necessary)
            sauvola: apply sauvola binarization
    text_standardize: preprocess and standardize sentence
"""

import os
import cv2
import numpy as np

from dpeter.constants import WHITE_CONSTANT


def adjust_to_see(img):
    """Rotate and transpose to image visualize (cv2 method or jupyter notebook)"""

    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), -90, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    img = cv2.warpAffine(img, M, (nW + 1, nH + 1))
    img = cv2.warpAffine(img.transpose(), M, (nW, nH))

    return img


def augmentation(imgs,
                 rotation_range=0,
                 scale_range=0,
                 height_shift_range=0,
                 width_shift_range=0,
                 dilate_range=1,
                 erode_range=1):
    """Apply variations to a list of images (rotate, width and height shift, scale, erode, dilate)"""

    imgs = imgs.astype(np.float32)
    _, h, w = imgs.shape

    dilate_kernel = np.ones((int(np.random.uniform(1, dilate_range)),), np.uint8)
    erode_kernel = np.ones((int(np.random.uniform(1, erode_range)),), np.uint8)
    height_shift = np.random.uniform(-height_shift_range, height_shift_range)
    rotation = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(1 - scale_range, 1)
    width_shift = np.random.uniform(-width_shift_range, width_shift_range)

    trans_map = np.float32([[1, 0, width_shift * w], [0, 1, height_shift * h]])
    rot_map = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)

    trans_map_aff = np.r_[trans_map, [[0, 0, 1]]]
    rot_map_aff = np.r_[rot_map, [[0, 0, 1]]]
    affine_mat = rot_map_aff.dot(trans_map_aff)[:2, :]

    for i in range(len(imgs)):
        imgs[i] = cv2.warpAffine(imgs[i], affine_mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=255)
        imgs[i] = cv2.erode(imgs[i], erode_kernel, iterations=1)
        imgs[i] = cv2.dilate(imgs[i], dilate_kernel, iterations=1)

    return imgs


def normalization(imgs):
    """Normalize list of images"""

    imgs = np.asarray(imgs).astype(np.float32)
    imgs = np.expand_dims(imgs / 255, axis=-1)
    return imgs


"""
Preprocess metodology based in:
    H. Scheidl, S. Fiel and R. Sablatnig,
    Word Beam Search: A Connectionist Temporal Classification Decoding Algorithm, in
    16th International Conference on Frontiers in Handwriting Recognition, pp. 256-258, 2018.
"""


def rotate_maybe(img: np.ndarray) -> np.ndarray:
    w, h = img.shape
    if w > h * 2:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return img


def preprocess(path, input_size):
    """Make the process with the `input_size` to the scale resize"""

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = rotate_maybe(img)

    wt, ht, _ = input_size
    h, w = np.array(img).shape
    f = max((w / wt), (h / ht))

    new_size = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1))
    img = cv2.resize(img, new_size)

    target = np.full([ht, wt], fill_value=WHITE_CONSTANT, dtype=np.uint8)
    target[0:new_size[1], 0:new_size[0]] = img
    img = cv2.transpose(target)

    return img


def text_standardize(text):
    """Organize/add spaces around punctuation marks"""

    if text is None:
        return ""

    return text.strip()
