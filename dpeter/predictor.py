from typing import Dict

from allennlp.predictors import Predictor
from allennlp.data.instance import Instance

from dpeter.utils.data import load_image, load_text


class PeterPredictor(Predictor):
    def _json_to_instance(self, json_dict: Dict[str, str]) -> Instance:
        image_path = json_dict["image_path"]
        image = load_image(image_path)

        text_path = json_dict.get("text_path")
        if text_path is not None:
            text = load_text(text_path)
        else:
            text = None
        return self._dataset_reader.text_to_instance(image, text)
