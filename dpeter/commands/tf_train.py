from typing import Optional, List
from pathlib import Path
import datetime

import typer
import wandb
from wandb.keras import WandbCallback
import editdistance
import numpy as np
from allennlp.common import Params

from dpeter.constants import PROJECT_NAME, CHARSET, MAX_LENGTH, INPUT_SIZE, BENTAM_CHARSET
from dpeter.utils.generator import DataGenerator
from dpeter.models.htr_model import HTRModel
from dpeter.utils.metrics import ocr_metrics
from dpeter.utils.data import load_jsonlines, load_images, load_texts
from dpeter.utils.preprocessing import rotate_maybe
from dpeter.utils.augmentators.augmentator import Augmentator
from dpeter.utils.postprocessors.postprocessor import Postprocessor

app = typer.Typer()

NUM_SAMPLES = 50


def find_indexes_of_the_worst_predicitons(y_pred: List[str], y_true: List[str], k: int = NUM_SAMPLES) -> np.ndarray:

    dists = []
    for yt, yp in zip(y_true, y_pred):
        dists.append(editdistance.eval(yt, yp))
    dists = np.array(dists)
    indexes = dists.argsort()[::-1][:k]
    return indexes


@app.command()
def main(config_path: Path, data_dir: Path, serialization_dir: Optional[Path] = None):
    wandb.init(project=PROJECT_NAME)

    if serialization_dir is None:
        date = datetime.datetime.utcnow().strftime('%H%M%S-%d%m')
        serialization_dir = Path(f'./logs/{date}')

    serialization_dir.mkdir(exist_ok=True, parents=True)

    params = Params.from_file(str(config_path))
    params["data_dir"] = str(data_dir)
    params["serialization_dir"] = str(serialization_dir)
    save_config_to = str(serialization_dir / "config.json")
    params.to_file(save_config_to)
    wandb.save(save_config_to)

    flat_params = params.as_flat_dict()
    wandb.config.update(flat_params)

    source_path = data_dir / "data.hdf5"

    if params["is_bentam"]:
        charset = BENTAM_CHARSET
        maxlen = 110
    else:
        charset = CHARSET
        maxlen = MAX_LENGTH

    dtgen = DataGenerator(
        source=str(source_path),
        batch_size=params["training"]["batch_size"],
        charset=charset,
        max_text_length=maxlen,
        augmentator=Augmentator.from_params(params["dataset_reader"]["augmentator"]),
        predict=False,
        train_flips=params.get("train_flips", False),
    )

    model = HTRModel(
        architecture=params["model"]["type"],
        input_size=INPUT_SIZE,
        vocab_size=dtgen.tokenizer.vocab_size,
        beam_width=params["model"]["beam_size"],
        stop_tolerance=params["training"]["patience"],
        reduce_tolerance=params["training"]["lr_patience"],
        train_flips=params.get("train_flips", False),
    )

    model.compile(learning_rate=params["training"]["learning_rate"])
    model.load_checkpoint('presets/bentam.hdf5')
    checkpoint_path = str(serialization_dir / "checkpoint_weights.hdf5")
    # can be used as a pretrained model (save for later)
    # model.load_checkpoint(target=checkpoint_path)

    model.summary(str(serialization_dir), "summary.txt")
    callbacks = model.get_callbacks(
        logdir=str(serialization_dir),
        checkpoint=checkpoint_path,
        verbose=1
    ) + [WandbCallback(save_model=False)]

    start_time = datetime.datetime.now()

    h = model.fit(
        x=dtgen.next_train_batch(),
        epochs=params["training"]["num_epochs"],
        steps_per_epoch=dtgen.steps['train'],
        validation_data=dtgen.next_valid_batch(),
        validation_steps=dtgen.steps['valid'],
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )
    model.load_checkpoint(target=checkpoint_path)

    wandb.save(checkpoint_path)

    total_time = datetime.datetime.now() - start_time

    loss = h.history['loss']
    val_loss = h.history['val_loss']

    min_val_loss = min(val_loss)
    min_val_loss_i = val_loss.index(min_val_loss)

    time_epoch = (total_time / len(loss))
    total_item = (dtgen.size['train'] + dtgen.size['valid'])

    t_corpus = "\n".join([
        f"Total train images:      {dtgen.size['train']}",
        f"Total validation images: {dtgen.size['valid']}",
        f"Batch:                   {dtgen.batch_size}\n",
        f"Total time:              {total_time}",
        f"Time per epoch:          {time_epoch}",
        f"Time per item:           {time_epoch / total_item}\n",
        f"Total epochs:            {len(loss)}",
        f"Best epoch               {min_val_loss_i + 1}\n",
        f"Training loss:           {loss[min_val_loss_i]:.8f}",
        f"Validation loss:         {min_val_loss:.8f}"
    ])

    with open(str(serialization_dir / "train.txt"), "w") as lg:
        lg.write(t_corpus)
        print(t_corpus)

    # MAKE SOME PREDICTIONS
    predicts, _ = model.predict(
        x=dtgen.next_valid_batch(),
        steps=dtgen.steps['valid'],
        ctc_decode=True,
        verbose=1,
    )
    predicts = [dtgen.tokenizer.decode(x[0]) for x in predicts]

    postprocessor = Postprocessor.from_params(params["postprocessor"])
    predicts = postprocessor.postprocess(predicts)

    metrics = ocr_metrics(
        predicts=predicts,
        ground_truth=dtgen.dataset['valid']['gt'],
        norm_accentuation=False,
        norm_punctuation=False
    )
    metrics = {'CER': metrics[0], "WER": metrics[1], "SER": metrics[2]}
    print(">>> METRICS:")
    print(metrics)
    for key, val in metrics.items():
        wandb.run.summary[key] = val

    valid_data = load_jsonlines(str(data_dir / "valid.json"))
    sample_names = [Path(p['image_path']).stem for p in valid_data]
    images = load_images(valid_data)
    images = [rotate_maybe(img) for img in images]
    texts = load_texts(valid_data)
    wandb.log(
        {
            "examples": [
                wandb.Image(img, caption=f"{name}\ntrue = {text}\npred = {pred}")
                for img, pred, text, name in zip(
                    images[:NUM_SAMPLES], predicts[:NUM_SAMPLES], texts[:NUM_SAMPLES], sample_names[:NUM_SAMPLES]
                )
            ]
        }
    )

    worst_indexes = find_indexes_of_the_worst_predicitons(predicts, texts)
    wandb.log(
        {
            "worst_examples": [
                wandb.Image(images[idx], caption=f"{sample_names[idx]}\ntrue = {texts[idx]}\npred = {predicts[idx]}")
                for idx in worst_indexes
            ]
        }
    )


if __name__ == "__main__":
    app()
