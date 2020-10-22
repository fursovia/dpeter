from typing import Dict, Optional, Any

import torch
import numpy as np
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models.model import Model
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn.regularizers import RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, get_token_ids_from_text_field_tensors
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN

from dpeter.models.inception import get_inception_encoder
from dpeter.modules.metrics import CompetitionMetric
from dpeter.modules.spatial_attention import SpatialAttention
from dpeter.constants import END_TOKEN, START_TOKEN
from dpeter.utils.data import decode_indexes


@Model.register("img2sentence")
class Img2Sentence(Model):

    MAX_SENTENCE_LENGTH = 128
    NUM_CHANNELS = 288

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
        self._embedder = torch.nn.Embedding(
            # TODO: can be smaller
            num_embeddings=self.vocab.get_vocab_size("tokens"),
            embedding_dim=self._emb_dim  # TODO
        )
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

        self._padding_index = self.vocab.get_token_index(DEFAULT_PADDING_TOKEN)
        self._start_index = self.vocab.get_token_index(START_TOKEN)
        self._end_index = self.vocab.get_token_index(END_TOKEN)
        self._unk_index = self.vocab.get_token_index(DEFAULT_OOV_TOKEN)
        self._space_index = self.vocab.get_token_index(" ")

        weight = torch.ones(self.vocab.get_vocab_size("tokens"))
        weight[self._space_index] = 2.0
        self._loss = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=self._padding_index)
        self._metric = CompetitionMetric(self.vocab)

    def _get_mask_from_length(self, length: torch.Tensor) -> torch.Tensor:
        length_numpy = length.cpu().numpy()
        max_length = length_numpy.max() + 2  # we also add start/end tokens
        mask = np.zeros(shape=(length.size(0), max_length))

        for i, curr_length in enumerate(length_numpy):
            mask[i, :(curr_length + 2)] = 1

        return torch.tensor(mask, dtype=torch.bool, device=length.device)

    def _get_embeddings_from_length(self, length: torch.Tensor) -> torch.Tensor:
        max_length = length.max().item() + 2  # we also add start/end tokens
        token_ids = torch.full(
            size=(length.size(0), max_length),
            fill_value=self._padding_index,
            dtype=torch.long,
            device=length.device
        )
        # can i do it in a vector manner?
        for i, curr_length in enumerate(length.cpu().numpy()):
            token_ids[i, :curr_length] = self._unk_index
        token_ids[:, 0] = self._start_index
        token_ids[:, length + 1] = self._end_index
        embeddings = self._embedder(token_ids)
        return embeddings

    def forward(
            self,
            image: torch.Tensor,
            text: Optional[TextFieldTensors] = None,
            length: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Dict[str, torch.Tensor]:
        batch_size = image.size(0)

        if len(image.shape) != 4:
            # we add one dimension
            image = torch.repeat_interleave(image.unsqueeze(1), 3, 1)
        else:
            image = image.transpose(1, 3)

        # (batch_size, 288, 13, 125)
        features = self._encoder(image)
        # (batch_size, 125, 13, 288)
        features = features.transpose(1, 3)

        if length is not None:
            mask = get_text_field_mask(text)
        else:
            with torch.no_grad():
                length = self._length_classifier(image)
                length = torch.clamp_min(length, 1).type(torch.long)
                length = length.to(image.device)

            mask = self._get_mask_from_length(length)

        max_length = length.max().item() + 2
        position_ids = torch.arange(max_length).unsqueeze(0).expand([batch_size, max_length])
        position_ids = position_ids.to(image.device)
        positional_embeddigns = self._positional_embedder(position_ids)

        # START/END/UNK/PADDING embeddings
        embeddings = self._get_embeddings_from_length(length)

        embeddings = positional_embeddigns + embeddings

        # (batch_size, num_tokens, 1625)
        attention_weights = self._attention(embeddings, features)
        # (batch_size, num_tokens, NUM_CHANNELS)
        attention_embeddings = self._attention.get_attention_features(features, attention_weights)

        # (batch_size, input_dim)
        embeddings = self._wc(embeddings) + self._wu1(attention_embeddings)

        # (batch_size, hidden_dim)
        contextual_embeddings = self._seq2seq_encoder(embeddings, mask=mask)

        # (batch_size, vocab_size)
        logits = self._wo(contextual_embeddings) + self._wu2(attention_embeddings)
        probs = torch.softmax(logits, dim=-1)
        top_indices = probs.argmax(dim=-1)

        output_dict = {"probs": probs, "top_indices": top_indices}
        # import pdb; pdb.set_trace()
        if text is not None:
            output_dict["loss"] = self._loss(
                logits.transpose(1, 2),
                get_token_ids_from_text_field_tensors(text),
            )

        if not self.training and text is not None:
            self._metric(predictions=top_indices, gold_labels=get_token_ids_from_text_field_tensors(text))

        return output_dict

    def get_metrics(self, reset: bool = False):
        return self._metric.get_metric(reset)

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        output_dict["sentences"] = decode_indexes(indexes=output_dict["top_indices"], vocab=self.vocab)
        return output_dict
