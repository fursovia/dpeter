import numpy as np
from allennlp.common import Registrable


class ImageAugmentator(Registrable):

    def __call__(self, image: np.ndarray) -> np.ndarray:
        pass


@ImageAugmentator.register("null")
class NullAugmentator(ImageAugmentator):

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return image