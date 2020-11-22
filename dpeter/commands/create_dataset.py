from pathlib import Path

import typer
import h5py
from allennlp.common import Params

from dpeter.utils.dataset import Dataset
from dpeter.utils.preprocessors.preprocessor import Preprocessor

app = typer.Typer()


@app.command()
def main(config_path: Path, data_dir: str = "./data"):
    params = Params.from_file(str(config_path))
    preprocessor = Preprocessor.from_params(params["dataset_reader"]["preprocessor"])

    ds = Dataset(data_dir=data_dir, preprocessor=preprocessor, bertam=params.get("is_bentam", False))
    ds.read()

    print("Partitions will be saved...")
    for partition in ds.partitions:
        with h5py.File(str(Path(data_dir) / "data.hdf5"), "a") as hf:
            hf.create_dataset(f"{partition}/dt", data=ds.dataset[partition]['dt'], compression="gzip", compression_opts=9)
            hf.create_dataset(f"{partition}/gt", data=ds.dataset[partition]['gt'], compression="gzip", compression_opts=9)
            print(f"[OK] {partition} partition.")

    print(f"Transformation finished.")


if __name__ == "__main__":
    app()
