import numpy as np
import cv2

from dpeter.utils.augmentators.augmentator import Augmentator
from dpeter.constants import WHITE_CONSTANT


@Augmentator.register("flor")
class Flor(Augmentator):

    def __init__(
            self,
            dilate_range: float = 3,
            erode_range: float = 5,
            height_shift_range: float = 0.025,
            scale_range: float = 0.05,
            width_shift_range: float = 0.05,
            rotation_range: float = 1.5
    ):
        self.dilate_range = dilate_range
        self.erode_range = erode_range
        self.height_shift_range = height_shift_range
        self.scale_range = scale_range
        self.width_shift_range = width_shift_range
        self.rotation_range = rotation_range

    def augment(self, images: np.ndarray) -> np.ndarray:
        imgs = images.astype(np.float32)
        _, h, w = imgs.shape

        dilate_kernel = np.ones((int(np.random.uniform(1, self.dilate_range)),), np.uint8)
        erode_kernel = np.ones((int(np.random.uniform(1, self.erode_range)),), np.uint8)
        height_shift = np.random.uniform(-self.height_shift_range, self.height_shift_range)
        rotation = np.random.uniform(-self.rotation_range, self.rotation_range)
        scale = np.random.uniform(1 - self.scale_range, 1)
        width_shift = np.random.uniform(-self.width_shift_range, self.width_shift_range)

        trans_map = np.float32([[1, 0, width_shift * w], [0, 1, height_shift * h]])
        rot_map = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)

        trans_map_aff = np.r_[trans_map, [[0, 0, 1]]]
        rot_map_aff = np.r_[rot_map, [[0, 0, 1]]]
        affine_mat = rot_map_aff.dot(trans_map_aff)[:2, :]

        for i in range(len(imgs)):
            imgs[i] = cv2.warpAffine(imgs[i], affine_mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=WHITE_CONSTANT)
            imgs[i] = cv2.erode(imgs[i], erode_kernel, iterations=1)
            imgs[i] = cv2.dilate(imgs[i], dilate_kernel, iterations=1)

        return imgs
