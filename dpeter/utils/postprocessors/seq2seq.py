import json
from pathlib import Path
import os
import shutil

from allennlp_models.generation.predictors import Seq2SeqPredictor
from allennlp.models.archival import _load_dataset_readers, _load_model, get_weights_path, Archive, load_archive, extracted_archive
from allennlp.common import Params
from allennlp.common.file_utils import cached_path

from dpeter.utils.postprocessors.postprocessor import Postprocessor


@Postprocessor.register("seq2seq")
class Seq2seqPostprocessor(Postprocessor):

    # TODO: add batches
    def __init__(self, archive_path: str, beam_size: int = 10, cuda_device: int = 0) -> None:

        resolved_archive_path = cached_path(archive_path, cache_dir="presets")

        tempdir = None
        try:
            if os.path.isdir(resolved_archive_path):
                serialization_dir = resolved_archive_path
            else:
                with extracted_archive(resolved_archive_path, cleanup=False) as tempdir:
                    serialization_dir = tempdir

            weights_path = get_weights_path(serialization_dir)

            # Load config
            params = json.load(open(str(Path(serialization_dir) / "config.json")))
            params["model"]["beam_size"] = beam_size
            config = Params(params=params)

            # Instantiate model and dataset readers. Use a duplicate of the config, as it will get consumed.
            dataset_reader, validation_dataset_reader = _load_dataset_readers(
                config.duplicate(), serialization_dir
            )
            model = _load_model(config.duplicate(), weights_path, serialization_dir, cuda_device)
        finally:
            if tempdir is not None:
                shutil.rmtree(tempdir, ignore_errors=True)

        archive = Archive(
            model=model,
            config=config,
            dataset_reader=dataset_reader,
            validation_dataset_reader=validation_dataset_reader,
        )

        self._predictor = Seq2SeqPredictor.from_archive(archive)

    def postprocess(self, text: str) -> str:
        predictions = self._predictor.predict(text)
        pred_text = ''.join(predictions['predicted_tokens'][0])
        return pred_text
