import string
from typing import List
from spellchecker import SpellChecker
from multiprocessing import Pool

from dpeter.utils.postprocessors.postprocessor import Postprocessor


@Postprocessor.register("pyspell")
class PySpellChecker(Postprocessor):

    def __init__(
            self,
            corpus_path: str,
            distance: int = 2
    ):

        self.corpus = " ".join(open(corpus_path).read().splitlines()).lower()
        self.norvig = SpellChecker(language=None, distance=distance)
        self.norvig.word_frequency.load_words(self.corpus.split())

    def postprocess(self, texts: List[str]) -> List[str]:

        def fix_text(text):
            split = []
            for x in text.split():
                sugg = self.norvig.correction(x.lower()) if x not in string.punctuation else None
                split.append(sugg if sugg else x)

            return " ".join(split)

        with Pool(10) as p:
            predicts = p.map(fix_text, texts)

        return predicts
