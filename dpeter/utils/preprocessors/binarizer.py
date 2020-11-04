import numpy as np

from dpeter.utils.preprocessors.preprocessor import Preprocessor


@Preprocessor.register("null_binarizer")
class NullBinarizer(Preprocessor):

    """
    this is just an example
    """
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return image
