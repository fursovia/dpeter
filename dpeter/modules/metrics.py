from typing import Optional, Dict, Any

import torch
from allennlp.training.metrics import Metric
from allennlp.data.vocabulary import Vocabulary
import editdistance

from dpeter.utils.data import decode_indexes


class CompetitionMetric(Metric):

    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab
        self.cers = []
        self.wers = []
        self.hits = []

        self.clens = []
        self.wlens = []

    def __call__(
        self, predictions: torch.Tensor, gold_labels: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ):

        predicted_senteces = decode_indexes(predictions, self.vocab)
        true_senteces = decode_indexes(gold_labels, self.vocab)

        for ps, ts in zip(predicted_senteces, true_senteces):
            self.hits.append(ps == ts)
            self.cers.append(editdistance.eval(ps, ts))
            self.clens.append(len(ts))

            ps_words = ps.split()
            ts_words = ts.split()
            self.wers.append(editdistance.eval(ps_words, ts_words))
            self.wlens.append(len(ts_words))

    def get_metric(self, reset: bool) -> Dict[str, Any]:

        if self.wers:
            metrics = {
                "cer": sum(self.cers) / sum(self.clens),
                "wer": sum(self.wers) / sum(self.wlens),
                "acc": sum(self.hits) / len(self.hits),
            }
        else:
            metrics = {
                "cer": 0.0,
                "wer": 0.0,
                "acc": 0.0,
            }

        if reset:
            self.reset()

        return metrics

    def reset(self) -> None:
        self.cers = []
        self.wers = []
        self.hits = []
        self.clens = []
        self.wlens = []
