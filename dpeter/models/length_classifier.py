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

        inception = models.inception_v3()
        # Mixed_5d
        self._encoder = torch.nn.Sequential(*list(inception.children())[:10])
        # we assume (128, 1024) shape
        hidden_dim = 1625
        self._linear = torch.nn.Linear(hidden_dim, 1)
        self._loss = torch.nn.MSELoss()

    def forward(
            self,
            image: torch.Tensor,
            length: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Dict[str, torch.Tensor]:
        batch_size = image.size(0)

        if len(image.shape) != 4:
            # we add one dimension
            image = torch.repeat_interleave(image.unsqueeze(1), 3, 1)

        hidden = self._encoder(image)
        hidden = hidden.max(dim=1).values.reshape(batch_size, -1)
        y_pred = self._linear(hidden)

        output_dict = {"y_pred": y_pred, "hidden": hidden}
        if length is not None:
            output_dict["loss"] = self._loss(length.float(), y_pred.view(-1))

        return output_dict
