from typing import Dict, Optional

import torch

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn.regularizers import RegularizerApplicator


@Model.register("length_classifier")
class LengthClassifier(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            regularizer: RegularizerApplicator = None,
    ) -> None:
        super().__init__(vocab, regularizer)

        self._encoder: torch.nn.Module = None
        self._linear = torch.nn.Linear(self._encoder.get_output_dim(), 1)
        self._loss = torch.nn.MSELoss()

    def forward(
            self,
            image: torch.Tensor,
            length: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Dict[str, torch.Tensor]:

        hidden = self._encoder(image)
        y_pred = self._linear(hidden)

        output_dict = {"y_pred": y_pred, "hidden": hidden}
        if length is not None:
            output_dict["loss"] = self._loss(length.float(), y_pred.view(-1))

        return output_dict
