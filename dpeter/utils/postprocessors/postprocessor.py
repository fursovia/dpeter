from typing import List

from allennlp.common import Registrable


class Postprocessor(Registrable):

    def postprocess(self, texts: List[str]) -> List[str]:
        pass
