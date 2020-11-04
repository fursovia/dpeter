from allennlp.common import Registrable
import numpy as np


class Augmentator(Registrable):

    def augment(self, images: np.ndarray) -> np.ndarray:
        pass
