from typing import List, Dict
from pathlib import Path

from allennlp.models.archival import load_archive

import dpeter
from dpeter.predictor import PeterPredictor


def predict(serialization_dir: Path, data: List[Dict[str, str]], cuda_device: int = 0):
    archive = load_archive(str(serialization_dir / 'model.tar.gz'), cuda_device=cuda_device)
    predictor = PeterPredictor.from_archive(archive)
    preds = predictor.predict_batch_json(data)
    preds = [p['sentences'] for p in preds]
    return preds
