import numpy as np
import os
import cv2
from tensorflow import keras
from dpeter.utils.preprocessors.preprocessor import Preprocessor
from dpeter.models.htr_model import HTRModel
from dpeter.constants import INPUT_SIZE


@Preprocessor.register("basic_rotator")
class Rotator(Preprocessor):

    def __init__(self, threshold: float = 2.0) -> None:
        self._threshold = threshold

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        w, h = image.shape
        if w > h * self._threshold:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image


@Preprocessor.register("dropper")
class Dropper(Preprocessor):

    def __init__(self, threshold: float = 2.0) -> None:
        self._threshold = threshold

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        w, h = image.shape
        if w > h * self._threshold:
            return None
        return image


@Preprocessor.register("smart_flipper")
class Flipper(Preprocessor):

    def __init__(self, threshold: float = 2.0) -> None:
        self._threshold = threshold
        self._model = HTRModel(
            architecture="flor",
            input_size=INPUT_SIZE,
            vocab_size=78,
            beam_width=1,
            top_paths=1,
            train_flips=True,
        )
        self._model.load_checkpoint('presets/flipper.hdf5')

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        res = self._model.predict(image[None])
        if res < 0.5:
            return image[:, ::-1]
        return image