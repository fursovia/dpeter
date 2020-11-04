from typing import Tuple

import numpy as np
import cv2

from dpeter.utils.preprocessors.preprocessor import Preprocessor
from dpeter.constants import INPUT_SIZE, WHITE_CONSTANT


@Preprocessor.register("basic_resizer")
class Resizer(Preprocessor):

    def __init__(self, image_size: Tuple[int, int, int] = INPUT_SIZE) -> None:
        self._image_size = image_size

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        wt, ht, _ = self._image_size
        h, w = np.array(image).shape
        f = max((w / wt), (h / ht))

        new_size = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1))
        image = cv2.resize(image, new_size)

        target = np.full([ht, wt], fill_value=WHITE_CONSTANT, dtype=np.uint8)
        target[0:new_size[1], 0:new_size[0]] = image
        image = cv2.transpose(target)
        return image
