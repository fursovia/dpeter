from typing import List, Optional
import itertools

import numpy as np

from dpeter.models.htr_model import HTRModel
from dpeter.utils.generator import Tokenizer
from dpeter.utils.postprocessors.postprocessor import Postprocessor


class TfPredictor:

    def __init__(
            self,
            model: HTRModel,
            tokenizer: Tokenizer,
            postprocessor: Optional[Postprocessor] = None,
            batch_size: int = 16
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._postprocessor = postprocessor
        self._batch_size = batch_size

    def predict(self, images: np.ndarray) -> List[List[str]]:
        predictions, probabilities = self._model.predict(
            images,
            batch_size=self._batch_size,
            ctc_decode=True,
            verbose=1,
            steps=int(np.ceil(len(images) / self._batch_size)),
            use_multiprocessing=True,
            workers=5,
        )
        predictions = [[self._tokenizer.decode(y) for y in x] for x in predictions]

        if self._postprocessor is not None:
            print("postprocessing using seq2seq ...")
            num_paths = len(predictions[0])
            unraveled_predictions = list(itertools.chain(*predictions))
            unraveled_predictions = self._postprocessor.postprocess(unraveled_predictions)

            raveled_preds = []
            for i in range(len(unraveled_predictions) // num_paths):
                example_preds = []
                for j in range(num_paths):
                    example_preds.append(unraveled_predictions[i + j])
                raveled_preds.append(example_preds)

            return raveled_preds

        return predictions
