local dataset_size = 5000;
local batch_size = 32;
local batches_per_epoch = std.parseInt(dataset_size / batch_size);

{
  "data_loader": {
    "shuffle": true,
    "batch_size": batch_size,
    "batches_per_epoch": batches_per_epoch,
    "num_workers": 20,
    // https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
    "pin_memory": true
  },
  "trainer": {
    "num_epochs": 100,
    "patience": 5,
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    },
    "epoch_callbacks": ["wandb"],
    "cuda_device": 0
  }
}
