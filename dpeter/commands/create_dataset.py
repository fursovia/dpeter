from pathlib import Path

import typer
import h5py

from dpeter.utils.dataset import Dataset
from dpeter.constants import WIDTH, HEIGHT

INPUT_SIZE = (WIDTH, HEIGHT, 1)
app = typer.Typer()


@app.command()
def main(data_dir: str = "./data"):
    ds = Dataset(data_dir=data_dir)
    ds.read_partitions()

    print("Partitions will be preprocessed...")
    ds.preprocess_partitions(input_size=INPUT_SIZE)

    print("Partitions will be saved...")
    for partition in ds.partitions:
        with h5py.File(str(Path(data_dir) / "data.hdf5"), "a") as hf:
            hf.create_dataset(f"{partition}/dt", data=ds.dataset[partition]['dt'], compression="gzip", compression_opts=9)
            hf.create_dataset(f"{partition}/gt", data=ds.dataset[partition]['gt'], compression="gzip", compression_opts=9)
            print(f"[OK] {partition} partition.")

    print(f"Transformation finished.")


if __name__ == "__main__":
    app()
