from typing import Optional
from datetime import datetime
from pathlib import Path
import json

import typer
from allennlp.commands.train import train_model
from allennlp.common import Params
import wandb

import dpeter

PROJECT_NAME = "digital_peter"
app = typer.Typer()


@app.command()
def train(param_path: Path, serialization_dir: Optional[Path] = None):
    wandb.init(project=PROJECT_NAME)

    params = Params.from_file(str(param_path))
    flat_params = params.as_flat_dict()
    wandb.config.update(flat_params)

    if serialization_dir is None:
        date = datetime.utcnow().strftime('%H%M%S-%d%m')
        serialization_dir = Path(f'.logs/{date}-{param_path.stem}')

    train_model(params, str(serialization_dir))

    # TODO: should I add intermediate metrics?
    with open(str(serialization_dir / "metrics.json")) as f:
        metrics = json.load(f)

    wandb.log(metrics)
    wandb.save(str(serialization_dir / "model.tar.gz"))


if __name__ == "__main__":
    app()
