from typing import Optional
from datetime import datetime
from pathlib import Path
import json

import typer
from allennlp.commands.train import train_model
from allennlp.common import Params
import wandb

import dpeter
from dpeter.commands.predict import predict
from dpeter.utils.data import load_jsonlines, load_images, load_texts


PROJECT_NAME = "digital_peter"
NUM_SAMPLES = 50
app = typer.Typer()


@app.command()
def train(param_path: Path, data_dir: Optional[Path] = None, serialization_dir: Optional[Path] = None):
    wandb.init(project=PROJECT_NAME)

    if data_dir is None:
        data_dir = Path("./data")

    params = Params.from_file(
        str(param_path),
        ext_vars={
            "TRAIN_DATA_PATH": str(data_dir / "train.json"),
            "VALID_DATA_PATH": str(data_dir / "valid.json")
        }
    )
    flat_params = params.as_flat_dict()
    wandb.config.update(flat_params)

    if serialization_dir is None:
        date = datetime.utcnow().strftime('%H%M%S-%d%m')
        serialization_dir = Path(f'./logs/{date}-{param_path.stem}')

    train_model(params, str(serialization_dir))

    with open(str(serialization_dir / "metrics.json")) as f:
        metrics = json.load(f)

        for key, val in metrics.items():
            wandb.run.summary[key] = val

    wandb.save(str(serialization_dir / "model.tar.gz"))

    if params["model"]["type"] == "img2sentence":
        valid_data = load_jsonlines(str(data_dir / "valid.json"))[:NUM_SAMPLES]
        preds = predict(serialization_dir, valid_data)
        images = load_images(valid_data)
        texts = load_texts(valid_data)

        wandb.log(
            {
                "examples": [
                    wandb.Image(img, caption=f"true = {text}\npred = {pred}")
                    for img, pred, text in zip(images, preds, texts)
                ]
            }
        )


if __name__ == "__main__":
    app()
