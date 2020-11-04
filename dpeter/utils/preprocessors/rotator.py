import numpy as np
import cv2

from dpeter.utils.preprocessors.preprocessor import Preprocessor


@Preprocessor.register("basic_rotator")
class Rotator(Preprocessor):

    def __init__(self, threshold: float = 2.0) -> None:
        self._threshold = threshold

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        w, h = image.shape
        if w > h * self._threshold:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image
