from typing import Optional, List
from pathlib import Path

import typer
import editdistance
import numpy as np

from dpeter.constants import CHARSET, MAX_LENGTH, INPUT_SIZE
from dpeter.utils.generator import Tokenizer
from dpeter.utils.preprocessing import preprocess, normalization
from dpeter.models.htr_model import HTRModel

app = typer.Typer()


NUM_SAMPLES = 50
ARCH = "flor"
BATCH_SIZE = 16
NUM_EPOCHS = 100


def find_indexes_of_the_worst_predicitons(y_pred: List[str], y_true: List[str], k: int = NUM_SAMPLES) -> np.ndarray:

    dists = []
    for yt, yp in zip(y_true, y_pred):
        dists.append(editdistance.eval(yt, yp))
    dists = np.array(dists)
    indexes = dists.argsort()[::-1][:k]
    return indexes


@app.command()
def main(serialization_dir: Path, data_dir: Optional[Path] = None, out_path: Optional[Path] = None,):

    if data_dir is None:
        data_dir = Path("./data")

    if out_path is None:
        out_path = Path("./output")

    tokenizer = Tokenizer(chars=CHARSET, max_text_length=MAX_LENGTH)

    model = HTRModel(
        architecture=ARCH,
        input_size=INPUT_SIZE,
        vocab_size=len(CHARSET) + 2,
        beam_width=10,
        stop_tolerance=20,
        reduce_tolerance=15,
        top_paths=1
    )

    model.compile()
    model.load_checkpoint(target=str(serialization_dir / "checkpoint_weights.hdf5"))

    images = []
    names = []
    for image_path in Path(data_dir).glob("*.jpg"):
        img = preprocess(str(image_path), INPUT_SIZE)
        images.append(img)
        names.append(image_path.stem)

    images = normalization(images)
    predicts, probabilities = model.predict(images, batch_size=BATCH_SIZE, ctc_decode=True)
    predicts = [tokenizer.decode(x[0]) for x in predicts]

    out_path.mkdir(exist_ok=True, parents=True)
    for name, predict in zip(names, predicts):
        with open(out_path / f"{name}.txt", "w") as f:
            f.write(predict)


if __name__ == "__main__":
    app()
