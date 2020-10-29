from typing import Optional, List
from pathlib import Path
import datetime

import typer
import wandb
import editdistance
import numpy as np

from dpeter.constants import WIDTH, HEIGHT, PROJECT_NAME
from dpeter.utils.generator import DataGenerator
from dpeter.models.htr_model import HTRModel
from dpeter.utils.metrics import ocr_metrics
from dpeter.utils.data import load_jsonlines, load_images, load_texts

app = typer.Typer()


INPUT_SIZE = (WIDTH, HEIGHT, 1)
NUM_SAMPLES = 50
ARCH = "flor"
BATCH_SIZE = 16
MAX_LENGTH = 80
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
CHARSET = " оаетнисрлвдкмпубiѣяычгзъжхшйюфц1ь+щ[0]27345э8erps96tfіhcn⊕m⊗daglb)–|×o/kuǂ…"


def find_indexes_of_the_worst_predicitons(y_pred: List[str], y_true: List[str], k: int = 50) -> np.ndarray:

    dists = []
    for yt, yp in zip(y_true, y_pred):
        dists.append(editdistance.eval(yt, yp))
    dists = np.array(dists)
    indexes = dists.argsort()[::-1][:k]
    return indexes


@app.command()
def main(data_dir: Path, serialization_dir: Optional[Path] = None):
    wandb.init(project=PROJECT_NAME)

    if serialization_dir is None:
        date = datetime.datetime.utcnow().strftime('%H%M%S-%d%m')
        serialization_dir = Path(f'./logs/{date}')

    source_path = data_dir / "data.hdf5"

    dtgen = DataGenerator(
        source=str(source_path),
        batch_size=BATCH_SIZE,
        charset=CHARSET,
        max_text_length=MAX_LENGTH,
        predict=False
    )

    model = HTRModel(
        architecture=ARCH,
        input_size=INPUT_SIZE,
        vocab_size=dtgen.tokenizer.vocab_size,
        beam_width=10,
        stop_tolerance=20,
        reduce_tolerance=15
    )

    model.compile(learning_rate=LEARNING_RATE)
    checkpoint_path = str(serialization_dir / "checkpoint_weights.hdf5")
    # model.load_checkpoint(target=checkpoint_path)

    model.summary(str(serialization_dir), "summary.txt")
    callbacks = model.get_callbacks(
        logdir=str(serialization_dir),
        checkpoint=checkpoint_path,
        verbose=1
    )

    start_time = datetime.datetime.now()

    h = model.fit(
        x=dtgen.next_train_batch(),
        epochs=NUM_EPOCHS,
        steps_per_epoch=dtgen.steps['train'],
        validation_data=dtgen.next_valid_batch(),
        validation_steps=dtgen.steps['valid'],
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )

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
        verbose=1
    )
    predicts = [dtgen.tokenizer.decode(x[0]) for x in predicts]
    metrics = ocr_metrics(
        predicts=predicts,
        ground_truth=dtgen.dataset['valid']['gt'],
        norm_accentuation=False,
        norm_punctuation=False
    )
    metrics = {'CER': metrics[0], "WER": metrics[1], "SER": metrics[2]}
    for key, val in metrics.items():
        wandb.run.summary[key] = val

    valid_data = load_jsonlines(str(data_dir / "valid.json"))
    images = load_images(valid_data)
    texts = load_texts(valid_data)
    wandb.log(
        {
            "examples": [
                wandb.Image(img, caption=f"true = {text}\npred = {pred}")
                for img, pred, text in zip(images[:NUM_SAMPLES], predicts[:NUM_SAMPLES], texts[:NUM_SAMPLES])
            ]
        }
    )

    worst_indexes = find_indexes_of_the_worst_predicitons(predicts, texts)
    wandb.log(
        {
            "worst_examples": [
                wandb.Image(images[idx], caption=f"true = {texts[idx]}\npred = {predicts[idx]}")
                for idx in worst_indexes
            ]
        }
    )


if __name__ == "__main__":
    app()
