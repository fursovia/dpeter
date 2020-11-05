from allennlp.common import Registrable


class Postprocessor(Registrable):

    def postprocess(self, text: str) -> str:
        pass
