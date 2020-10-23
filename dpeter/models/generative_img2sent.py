from typing import Dict, List, Tuple, Optional, Any

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell, LSTM

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import Attention
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util
from allennlp.nn.beam_search import BeamSearch


from dpeter.constants import END_TOKEN, START_TOKEN
from dpeter.modules.image_encoder import InceptionEncoder, ImageEncoder
from dpeter.modules.metrics import CompetitionMetric


@Model.register("generative_img2sentence")
class GenerativeImg2Sentence(Model):
    target_namespace = "tokens"

    def __init__(
        self,
        vocab: Vocabulary,
        max_decoding_steps: int,
        attention: Attention,
        encoder: ImageEncoder = InceptionEncoder(),
        beam_size: int = 1,
        target_embedding_dim: int = 64,
        scheduled_sampling_ratio: float = 0.0,
        target_decoder_layers: int = 1,
    ) -> None:
        super().__init__(vocab)
        self._target_decoder_layers = target_decoder_layers
        self._scheduled_sampling_ratio = scheduled_sampling_ratio

        self._start_index = self.vocab.get_token_index(START_TOKEN, self.target_namespace)
        self._end_index = self.vocab.get_token_index(END_TOKEN, self.target_namespace)

        self._max_decoding_steps = max_decoding_steps
        self._beam_search = BeamSearch(
            self._end_index, max_steps=max_decoding_steps, beam_size=beam_size
        )

        # Encodes the sequence of source embeddings into a sequence of hidden states.
        self._encoder = encoder

        num_classes = self.vocab.get_vocab_size(self.target_namespace)

        # Attention mechanism applied to the encoder output for each step.
        self._attention = attention

        # Dense embedding of vocab words in the target space.
        self._target_embedder = Embedding(
            embedding_dim=target_embedding_dim,
            vocab_namespace=self.target_namespace,
            vocab=self.vocab,
        )

        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with the final hidden state of the encoder.
        self._encoder_output_dim = self._encoder.get_output_dim()
        self._decoder_output_dim = self._encoder_output_dim

        self._decoder_input_dim = self._decoder_output_dim + target_embedding_dim

        if self._target_decoder_layers > 1:
            self._decoder_cell = LSTM(
                self._decoder_input_dim,
                self._decoder_output_dim,
                self._target_decoder_layers,
            )
        else:
            self._decoder_cell = LSTMCell(self._decoder_input_dim, self._decoder_output_dim)

        self._output_projection_layer = Linear(self._decoder_output_dim, num_classes)
        self._metric = CompetitionMetric(self.vocab)

    def take_step(
        self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor], step: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # shape: (group_size, num_classes)
        output_projections, state = self._prepare_output_projections(last_predictions, state)

        # shape: (group_size, num_classes)
        class_log_probabilities = F.log_softmax(output_projections, dim=-1)

        return class_log_probabilities, state

    @overrides
    def forward(
            self,
            image: torch.Tensor,
            text: Optional[TextFieldTensors] = None,
            **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # source_mask, encoder_outputs
        state = self._encode(image)

        if text:
            state = self._init_decoder_state(state)
            output_dict = self._forward_loop(state, text)
        else:
            output_dict = {}

        if not self.training:
            state = self._init_decoder_state(state)
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)

            if text is not None:
                top_k_predictions = output_dict["predictions"]
                # shape: (batch_size, max_predicted_sequence_length)
                best_predictions = top_k_predictions[:, 0, :]
                self._metric(predictions=best_predictions, gold_labels=text["tokens"]["tokens"])

        return output_dict

    def _encode(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._encoder(image)
        batch_size = encoder_outputs.size(0)
        max_input_sequence_length = encoder_outputs.size(1)
        source_mask = torch.ones(
            batch_size,
            max_input_sequence_length,
            dtype=torch.bool,
            device=image.device
        )
        return {"source_mask": source_mask, "encoder_outputs": encoder_outputs}

    def _init_decoder_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = state["source_mask"].size(0)
        # shape: (batch_size, num_channels)
        final_encoder_output = torch.mean(state["encoder_outputs"], dim=1)

        # Initialize the decoder hidden state with the final output of the encoder.
        # shape: (batch_size, decoder_output_dim)
        state["decoder_hidden"] = final_encoder_output
        # shape: (batch_size, decoder_output_dim)
        state["decoder_context"] = state["encoder_outputs"].new_zeros(
            batch_size, self._decoder_output_dim
        )
        if self._target_decoder_layers > 1:
            # shape: (num_layers, batch_size, decoder_output_dim)
            state["decoder_hidden"] = (
                state["decoder_hidden"].unsqueeze(0).repeat(self._target_decoder_layers, 1, 1)
            )

            # shape: (num_layers, batch_size, decoder_output_dim)
            state["decoder_context"] = (
                state["decoder_context"].unsqueeze(0).repeat(self._target_decoder_layers, 1, 1)
            )

        return state

    def _forward_loop(
        self, state: Dict[str, torch.Tensor], text: TextFieldTensors = None
    ) -> Dict[str, torch.Tensor]:

        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]
        batch_size = source_mask.size(0)

        if text:
            # shape: (batch_size, max_target_sequence_length)
            targets = text["tokens"]["tokens"]

            target_sequence_length = targets.size(1)
            num_decoding_steps = target_sequence_length - 1
        else:
            num_decoding_steps = self._max_decoding_steps

        # shape: (batch_size,)
        last_predictions = source_mask.new_full(
            (batch_size,), fill_value=self._start_index, dtype=torch.long
        )

        step_logits: List[torch.Tensor] = []
        step_predictions: List[torch.Tensor] = []
        for timestep in range(num_decoding_steps):
            if self.training and torch.rand(1).item() < self._scheduled_sampling_ratio:
                # Use gold tokens at test time and at a rate of 1 - _scheduled_sampling_ratio
                # during training.
                # shape: (batch_size,)
                input_choices = last_predictions
            elif not text:
                # shape: (batch_size,)
                input_choices = last_predictions
            else:
                # shape: (batch_size,)
                input_choices = targets[:, timestep]

            # shape: (batch_size, num_classes)
            output_projections, state = self._prepare_output_projections(input_choices, state)

            # list of tensors, shape: (batch_size, 1, num_classes)
            step_logits.append(output_projections.unsqueeze(1))

            # shape: (batch_size, num_classes)
            class_probabilities = F.softmax(output_projections, dim=-1)

            # shape (predicted_classes): (batch_size,)
            _, predicted_classes = torch.max(class_probabilities, 1)

            # shape (predicted_classes): (batch_size,)
            last_predictions = predicted_classes

            step_predictions.append(last_predictions.unsqueeze(1))

        # shape: (batch_size, num_decoding_steps)
        predictions = torch.cat(step_predictions, 1)

        output_dict = {"predictions": predictions}

        if text:
            # shape: (batch_size, num_decoding_steps, num_classes)
            logits = torch.cat(step_logits, 1)

            # Compute loss.
            target_mask = util.get_text_field_mask(text)
            loss = self._get_loss(logits, targets, target_mask)
            output_dict["loss"] = loss

        return output_dict

    def _forward_beam_search(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Make forward pass during prediction using a beam search."""
        batch_size = state["source_mask"].size()[0]
        start_predictions = state["source_mask"].new_full(
            (batch_size,), fill_value=self._start_index, dtype=torch.long
        )

        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        all_top_k_predictions, log_probabilities = self._beam_search.search(
            start_predictions, state, self.take_step
        )

        output_dict = {
            "class_log_probabilities": log_probabilities,
            "predictions": all_top_k_predictions,
        }
        return output_dict

    def _prepare_output_projections(
        self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # shape: (group_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]

        # shape: (group_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # shape: (num_layers, group_size, decoder_output_dim)
        decoder_hidden = state["decoder_hidden"]

        # shape: (num_layers, group_size, decoder_output_dim)
        decoder_context = state["decoder_context"]

        # shape: (group_size, target_embedding_dim)
        embedded_input = self._target_embedder(last_predictions)

        # shape: (group_size, encoder_output_dim)
        if self._target_decoder_layers > 1:
            attended_input = self._prepare_attended_input(
                decoder_hidden[0], encoder_outputs, source_mask
            )
        else:
            attended_input = self._prepare_attended_input(
                decoder_hidden, encoder_outputs, source_mask
            )
        # shape: (group_size, decoder_output_dim + target_embedding_dim)
        decoder_input = torch.cat((attended_input, embedded_input), -1)

        if self._target_decoder_layers > 1:
            # shape: (1, batch_size, target_embedding_dim)
            decoder_input = decoder_input.unsqueeze(0)

            # shape (decoder_hidden): (num_layers, batch_size, decoder_output_dim)
            # shape (decoder_context): (num_layers, batch_size, decoder_output_dim)
            with torch.cuda.amp.autocast(False):
                _, (decoder_hidden, decoder_context) = self._decoder_cell(
                    decoder_input.float(), (decoder_hidden.float(), decoder_context.float())
                )
        else:
            # shape (decoder_hidden): (batch_size, decoder_output_dim)
            # shape (decoder_context): (batch_size, decoder_output_dim)
            with torch.cuda.amp.autocast(False):
                decoder_hidden, decoder_context = self._decoder_cell(
                    decoder_input.float(), (decoder_hidden.float(), decoder_context.float())
                )

        state["decoder_hidden"] = decoder_hidden
        state["decoder_context"] = decoder_context

        # shape: (group_size, num_classes)
        if self._target_decoder_layers > 1:
            output_projections = self._output_projection_layer(decoder_hidden[-1])
        else:
            output_projections = self._output_projection_layer(decoder_hidden)
        return output_projections, state

    def _prepare_attended_input(
        self,
        decoder_hidden_state: torch.LongTensor = None,
        encoder_outputs: torch.LongTensor = None,
        encoder_outputs_mask: torch.BoolTensor = None,
    ) -> torch.Tensor:
        """Apply attention over encoder outputs and decoder state."""
        # shape: (batch_size, max_input_sequence_length)
        input_weights = self._attention(decoder_hidden_state, encoder_outputs, encoder_outputs_mask)

        # shape: (batch_size, encoder_output_dim)
        attended_input = util.weighted_sum(encoder_outputs, input_weights)

        return attended_input

    @staticmethod
    def _get_loss(
        logits: torch.LongTensor,
        targets: torch.LongTensor,
        target_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        # shape: (batch_size, num_decoding_steps)
        relevant_targets = targets[:, 1:].contiguous()

        # shape: (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()

        return util.sequence_cross_entropy_with_logits(
            logits=logits,
            targets=relevant_targets,
            weights=relevant_mask,
            label_smoothing=None,
            gamma=None,
            alpha=None
        )

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return self._metric.get_metric(reset)

    def make_output_human_readable(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:

        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_tokens = []
        for top_k_predictions in predicted_indices:
            # Beam search gives us the top k results for each source sentence in the batch
            # we want top-k results.
            if len(top_k_predictions.shape) == 1:
                top_k_predictions = [top_k_predictions]

            batch_predicted_tokens = []
            for indices in top_k_predictions:
                indices = list(indices)
                # Collect indices till the first end_symbol
                if self._end_index in indices:
                    indices = indices[: indices.index(self._end_index)]
                predicted_tokens = [
                    self.vocab.get_token_from_index(x, namespace=self.target_namespace)
                    for x in indices
                ]
                batch_predicted_tokens.append(predicted_tokens)

            all_predicted_tokens.append(batch_predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict
