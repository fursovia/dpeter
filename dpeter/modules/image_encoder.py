from allennlp.common import Registrable
from torchvision import models
import torch


class ImageEncoder(torch.nn.Module, Registrable):

    def get_input_dim(self) -> int:
        """
        Returns the dimension of the vector input for each element in the sequence input
        to a `Seq2SeqEncoder`. This is `not` the shape of the input tensor, but the
        last element of that shape.
        """
        raise NotImplementedError

    def get_output_dim(self) -> int:
        """
        Returns the dimension of each vector in the sequence output by this `Seq2SeqEncoder`.
        This is `not` the shape of the returned tensor, but the last element of that shape.
        """
        raise NotImplementedError


class InceptionEncoder(ImageEncoder):

    def __init__(self):
        super().__init__()
        inception = models.inception_v3()
        modules = list(inception.children())[:10]
        self._encoder = torch.nn.Sequential(*modules)

    def get_output_dim(self):
        return 288

    def get_input_dim(self):
        return 3

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self._encoder(image.transpose(1, 3))
        features = features.transpose(1, 3)
        # 1625 tokens
        features = features.reshape(image.size(0), -1, self.get_output_dim())
        return features
