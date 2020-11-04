from allennlp.common import Registrable
import numpy as np


class Preprocessor(Registrable):

    """
    This guy is basically a module that is applied for training and validation (!) steps
    """

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        pass
