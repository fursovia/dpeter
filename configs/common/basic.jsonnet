{
  "data_loader": {
    "shuffle": false,
    "batch_size": 64,
    "num_workers": 10,
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
