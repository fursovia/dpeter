from typing import Dict, Optional

import torch
import numpy as np
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models.model import Model
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn.regularizers import RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN

from dpeter.models.inception import get_inception_encoder
from dpeter.modules.spatial_attention import SpatialAttention
from dpeter.reader import START_TOKEN, END_TOKEN


@Model.register("img2sentence")
class Img2Sentence(Model):

    MAX_SENTENCE_LENGTH = 128
    NUM_CHANNELS = 288
    PADDING_ID = 0

    def __init__(
            self,
            vocab: Vocabulary,
            seq2seq_encoder: Seq2SeqEncoder,
            length_classifier: Model,  # LengthClassifier
            input_dim: int = 128,
            emb_dim: int = 64,
            att_dim: int = 32,
            regularizer: RegularizerApplicator = None,
    ) -> None:
        super().__init__(vocab, regularizer)
        self._emb_dim = emb_dim
        self._att_dim = att_dim
        self._input_dim = input_dim
        # we have 71 max sentence length
        self._positional_embedder = torch.nn.Embedding(
            num_embeddings=self.MAX_SENTENCE_LENGTH,
            embedding_dim=self._emb_dim
        )
        self._encoder = get_inception_encoder()
        self._seq2seq_encoder = seq2seq_encoder
        self._length_classifier = length_classifier.eval()
        self._attention = SpatialAttention(
            num_channels=self.NUM_CHANNELS,
            embedding_dim=emb_dim,
            hidden_dim=self._att_dim
        )

        self._wc = torch.nn.Linear(self._emb_dim, self._input_dim)
        self._wu1 = torch.nn.Linear(self.NUM_CHANNELS, self._input_dim)

        self._wo = torch.nn.Linear(self._seq2seq_encoder.get_output_dim(), self.vocab.get_vocab_size("tokens"))
        self._wu2 = torch.nn.Linear(self.NUM_CHANNELS, self.vocab.get_vocab_size("tokens"))

        ignore_index = self.vocab.get_token_index(DEFAULT_PADDING_TOKEN)
        self._loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def _get_mask_from_length(self, length: torch.Tensor) -> torch.Tensor:
        length_numpy = length.cpu().numpy()
        max_length = length_numpy.max() + 2  # we also add start/end tokens
        mask = np.zeros(shape=(length.size(0), max_length))

        for i, curr_length in enumerate(length_numpy):
            mask[i, :(curr_length + 2)] = 1

        return torch.tensor(mask, dtype=torch.bool, device=length.device)

    def _get_embeddings_from_length(self, length: torch.Tensor) -> torch.Tensor:
        pass

    def forward(
            self,
            image: torch.Tensor,
            text: Optional[TextFieldTensors] = None,
            length: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # TODO: dont forget about start/end tokens
        batch_size = image.size(0)

        if len(image.shape) != 4:
            # we add one dimension
            image = torch.repeat_interleave(image.unsqueeze(1), 3, 1)

        # (batch_size, 288, 13, 125)
        features = self._encoder(image)
        # (batch_size, 125, 13, 288)
        features = features.transpose(1, 3)

        if text is not None and length is not None:
            mask = get_text_field_mask(text)
        else:
            with torch.no_grad():
                length = self._length_classifier(image)
                length = torch.clamp_min(length, 1).type(torch.long)

            mask = self._get_mask_from_length(length)

        max_length = length.max().item()
        position_ids = torch.arange(max_length).unsqueeze(0).expand([batch_size, max_length])
        positional_embeddigns = self._positional_embedding(position_ids)

        # TODO: do it right
        # START/END/UNK/PADDING embeddings
        embeddings = torch.rand(batch_size, max_length, self._emb_dim)

        embeddings = positional_embeddigns + embeddings

        # (batch_size, num_tokens, 1625)
        attention_weights = self._attention(embeddings, features)
        attention_embeddings = self._attention.get_attention_features(features, attention_weights)

        embeddings = self._wc(embeddings) + self._wu1(attention_embeddings)

        contextual_embeddings = self._seq2seq_encoder(embeddings, mask=mask)
        logits = self._wo(contextual_embeddings) + self._wu2(attention_embeddings)

        output_dict = {"logits": logits}

        if text is not None:
            output_dict["loss"] = self._loss(
                logits.transpose(1, 2),
                text["tokens"]["tokens"],
            )

        return output_dict
