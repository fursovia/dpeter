{
  "data_loader": {
    "shuffle": false,
    "batch_size": 64,
    "num_workers": 0,
    // https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
    "pin_memory": true
  },
  "trainer": {
    "num_epochs": 12,
    "patience": 1,
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    },
    "cuda_device": -1
  }
}
