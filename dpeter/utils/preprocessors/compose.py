from typing import List

import numpy as np

from dpeter.utils.preprocessors.preprocessor import Preprocessor


@Preprocessor.register("compose")
class ComposePreprocessor(Preprocessor):

    def __init__(self, preprocessors: List[Preprocessor]) -> None:
        self._preprocessors = preprocessors

    def preprocess(self, image: np.ndarray) -> np.ndarray:

        for preprocessor in self._preprocessors:
            image = preprocessor.preprocess(image)
        return image
