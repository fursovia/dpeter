from typing import Optional
from pathlib import Path
import json

import typer
from allennlp.common import Params

from dpeter.constants import CHARSET, MAX_LENGTH, INPUT_SIZE
from dpeter.utils.generator import Tokenizer
from dpeter.utils.preprocessing import normalization
from dpeter.utils.preprocessors.preprocessor import Preprocessor
from dpeter.utils.data import load_image
from dpeter.models.htr_model import HTRModel
from dpeter.tf_predictor import TfPredictor
from dpeter.utils.postprocessors.postprocessor import Postprocessor

app = typer.Typer()


@app.command()
def main(
        serialization_dir: Path,
        batch_size: int = 16,
        beam_size: int = 50,
        top_paths: int = 1,
        data_dir: Optional[Path] = None,
        out_path: Optional[Path] = None,
):

    if data_dir is None:
        data_dir = Path("./data/images")

    if out_path is None:
        out_path = Path("./output")

    tokenizer = Tokenizer(chars=CHARSET, max_text_length=MAX_LENGTH)

    config_path = str(serialization_dir / "config.json")
    params = json.load(open(str(config_path)))

    params = Params(params=params)
    model = HTRModel(
        architecture=params["model"]["type"],
        input_size=INPUT_SIZE,
        vocab_size=len(CHARSET) + 2,
        beam_width=beam_size,
        top_paths=top_paths
    )

    model.compile()
    model.load_checkpoint(target=str(serialization_dir / "checkpoint_weights.hdf5"))

    preprocessor = Preprocessor.from_params(params["dataset_reader"]["preprocessor"])
    images = []
    names = []
    for image_path in Path(data_dir).glob("*.jpg"):
        img = load_image(str(image_path))
        img = preprocessor.preprocess(img)
        images.append(img)
        names.append(image_path.stem)
    images = normalization(images)

    postprocessor = Postprocessor.from_params(params["postprocessor"])
    predictor = TfPredictor(model=model, tokenizer=tokenizer, postprocessor=postprocessor, batch_size=batch_size)
    predicts = predictor.predict(images)

    out_path.mkdir(exist_ok=True, parents=True)
    for name, predict in zip(names, predicts):
        with open(out_path / f"{name}.txt", "w") as f:
            for i, pred in enumerate(predict):
                if i != 0:
                    f.write('\n')
                f.write(pred)


if __name__ == "__main__":
    app()
