from typing import List, Dict, Any, Sequence

import jsonlines
import torch
from allennlp.data import Vocabulary

from dpeter.constants import END_TOKEN


def load_jsonlines(path: str) -> List[Dict[str, Any]]:
    data = []
    with jsonlines.open(path, "r") as reader:
        for items in reader:
            data.append(items)
    return data


def write_jsonlines(data: Sequence[Dict[str, Any]], path: str) -> None:
    with jsonlines.open(path, "w") as writer:
        for ex in data:
            writer.write(ex)


def decode_indexes(
    indexes: torch.Tensor, vocab: Vocabulary, namespace="tokens",
) -> List[str]:
    indexes = indexes.cpu().numpy()

    sentences = []
    for curr_indexes in indexes:
        curr_sentence = []
        for idx in curr_indexes[1:]:  # start token is skipped
            token = vocab.get_token_from_index(idx, namespace=namespace)
            if token == END_TOKEN:
                break
            else:
                curr_sentence.append(token)

        sentences.append("".join(curr_sentence))

    return sentences
