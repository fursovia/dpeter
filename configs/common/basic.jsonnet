{
  "data_loader": {
    "shuffle": true,
    "batch_size": 256,
    "num_workers": 0,
    // https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
    "pin_memory": true
  },
  "trainer": {
    "num_epochs": 200,
    "patience": 10,
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    },
    "cuda_device": 0
  }
}
