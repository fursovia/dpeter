from typing import List, Optional

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
        predicts, probabilities = self._model.predict(
            images,
            batch_size=self._batch_size,
            ctc_decode=True,
            verbose=1,
            steps=int(np.ceil(len(images) / self._batch_size)),
            use_multiprocessing=True,
            workers=5,
        )
        predicts = [[self._tokenizer.decode(y) for y in x] for x in predicts]

        if self._postprocessor is not None:
            print("postprocessing using seq2seq ...")
            predicts = [p[0] for p in predicts]
            preds = self._postprocessor.postprocess(predicts)
            return preds

            # postprocessed_predicts = []
            # for preds in predicts:
            #     preds = self._postprocessor.postprocess(preds)
            #     postprocessed_predicts.append(preds)
            # return postprocessed_predicts

        return predicts
