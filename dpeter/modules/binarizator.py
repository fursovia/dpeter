import numpy as np
from allennlp.common import Registrable


class ImageBinarizator(Registrable):
    MAX_VALUE = 255
    MIN_VALUE = 0

    def __call__(self, image: np.ndarray) -> np.ndarray:
        pass


@ImageBinarizator.register("simple")
class SimpleBinarizator(ImageBinarizator):

    def __init__(self, alpha: float = 2.75, beta: float = -160.0, threshold: int = 220) -> None:
        super().__init__()
        self._alpha = alpha
        self._beta = beta
        self._threshold = threshold

    def __call__(self, image: np.ndarray) -> np.ndarray:
        denoised = self._alpha * image + self._beta

        img = np.clip(denoised, self.MIN_VALUE, self.MAX_VALUE).astype(np.uint8)
        cond = img >= self._threshold
        img[cond] = self.MAX_VALUE
        img[~cond] = self.MIN_VALUE
        return img
