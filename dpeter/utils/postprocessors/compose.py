from typing import List

from dpeter.utils.postprocessors.postprocessor import Postprocessor


@Postprocessor.register("compose")
class ComposePostprocessor(Postprocessor):

    def __init__(self, postprocessors: List[Postprocessor]) -> None:
        self._postprocessors = postprocessors

    def postprocess(self, texts: List[str]) -> List[str]:

        for postprocessor in self._postprocessors:
            texts = postprocessor.postprocess(texts)
        return texts
