from typing import Dict, Optional

import torch

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn.regularizers import RegularizerApplicator
import torchvision.models as models


@Model.register("length_classifier")
class LengthClassifier(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            regularizer: RegularizerApplicator = None,
    ) -> None:
        super().__init__(vocab, regularizer)

        self._encoder = models.mobilenet_v2(pretrained=True)
        hidden_dim = 1000  # mobilenet hidden dim
        self._linear = torch.nn.Linear(hidden_dim, 1)
        self._loss = torch.nn.MSELoss()

    def forward(
            self,
            image: torch.Tensor,
            length: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Dict[str, torch.Tensor]:

        hidden = self._encoder(image.transpose(1, 3))
        y_pred = self._linear(hidden)

        output_dict = {"y_pred": y_pred, "hidden": hidden}
        if length is not None:
            output_dict["loss"] = self._loss(length.float(), y_pred.view(-1))

        return output_dict
