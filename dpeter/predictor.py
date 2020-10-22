from typing import Dict

from allennlp.predictors import Predictor
from allennlp.data.instance import Instance
import cv2


class PeterPredictor(Predictor):
    def _json_to_instance(self, json_dict: Dict[str, str]) -> Instance:
        image_path = json_dict["image_path"]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        text_path = json_dict.get("text_path")
        if text_path is not None:
            with open(text_path) as f:
                text = f.read()
        else:
            text = None
        return self._dataset_reader.text_to_instance(image, text)
