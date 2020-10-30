from typing import List, Dict, Any, Sequence

import jsonlines
import torch
from allennlp.data import Vocabulary
import numpy as np
import cv2

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


def load_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image


def load_images(data: List[Dict[str, str]]) -> List[np.ndarray]:

    images = []
    for element in data:
        images.append(load_image(element["image_path"]))

    return images


def load_text(text_path: str) -> str:
    with open(text_path) as f:
        text = f.read()
    return text


def load_texts(data: List[Dict[str, str]]) -> List[str]:
    texts = []
    for element in data:
        texts.append(load_text(element["text_path"]))
    return texts
