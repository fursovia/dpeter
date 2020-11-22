from pathlib import Path
import tempfile
import string
from typing import List

from dpeter.utils.postprocessors.postprocessor import Postprocessor
from symspellpy.symspellpy import SymSpell


@Postprocessor.register("symspell")
class SymSpellChecker(Postprocessor):

    def __init__(
            self,
            corpus_path: str = "presets/corpus.txt",
            max_dictionary_edit_distance: int = 1,
            prefix_length: int = 12,
            count_threshold: int = 1
    ):

        dictionary_path = str(Path(tempfile.mkdtemp()) / "dictionary.txt")

        self.max_dictionary_edit_distance = max_dictionary_edit_distance
        self.symspell = SymSpell(
            max_dictionary_edit_distance=self.max_dictionary_edit_distance,
            prefix_length=prefix_length,
            count_threshold=count_threshold
        )
        self.symspell.create_dictionary(corpus_path)

        with open(dictionary_path, "w") as f:
            for key, count in self.symspell.words.items():
                f.write(f"{key} {count}\n")

        self.symspell.load_dictionary(dictionary_path, term_index=0, count_index=1)

    def postprocess(self, texts: List[str]) -> List[str]:
        predicts = []

        for i in range(len(texts)):
            split = []

            for x in texts[i].split():
                sugg = self.symspell.lookup(
                    x.lower(),
                    verbosity=0,
                    max_edit_distance=self.max_dictionary_edit_distance,
                    transfer_casing=True
                ) if x not in string.punctuation else None
                split.append(sugg[0].term if sugg else x)

            predicts.append(" ".join(split))

        return predicts
