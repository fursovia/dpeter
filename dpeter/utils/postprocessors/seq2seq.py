from allennlp_models.generation.predictors import Seq2SeqPredictor
from allennlp.models.archival import load_archive

from dpeter.utils.postprocessors.postprocessor import Postprocessor


@Postprocessor.register("seq2seq")
class Seq2seqPostprocessor(Postprocessor):

    # TODO: add batches
    def __init__(self, archive_path: str, beam_size: int = 10) -> None:
        archive = load_archive(archive_path, overrides={'model': {'beam_size': beam_size}})
        self._predictor = Seq2SeqPredictor.from_archive(archive)

    def postprocess(self, text: str) -> str:
        predictions = self._predictor.predict(text)
        pred_text = ''.join(predictions['predicted_tokens'][0])
        return pred_text
