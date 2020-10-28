from typing import Optional
from pathlib import Path
import datetime

import typer

from dpeter.constants import WIDTH, HEIGHT
from dpeter.utils.generator import DataGenerator
from dpeter.models.htr_model import HTRModel

app = typer.Typer()


INPUT_SIZE = (WIDTH, HEIGHT, 1)
ARCH = "flor"
BATCH_SIZE = 16
MAX_LENGTH = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
CHARSET = " оаетнисрлвдкмпубiѣяычгзъжхшйюфц1ь+щ[0]27345э8erps96tfіhcn⊕m⊗daglb)–|×o/kuǂ…"


@app.command()
def main(data_dir: Path, serialization_dir: Optional[Path] = None):
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


if __name__ == "__main__":
    app()
