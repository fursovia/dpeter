from typing import Optional
from datetime import datetime
from pathlib import Path
import json

import typer
from allennlp.commands.train import train_model
from allennlp.common import Params
import wandb

import dpeter

app = typer.Typer()


@app.command()
def predict(serialization_dir: str):
    pass


if __name__ == "__main__":
    app()
