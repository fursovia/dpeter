import numpy as np

from dpeter.utils.preprocessors.preprocessor import Preprocessor
from dpeter.constants import WHITE_CONSTANT


# should be applied after augmentation
@Preprocessor.register("normalizer")
class Normalizer(Preprocessor):

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        image = image / WHITE_CONSTANT
        return image


@Preprocessor.register("inverse_normalizer")
class InverseNormalizer(Preprocessor):

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        image = (image - WHITE_CONSTANT) / WHITE_CONSTANT
        return image
