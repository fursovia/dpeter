local COMMON = import 'common/basic.jsonnet';

local binarizator = "null";
local augmentator = "google";

local encoder_dim = 288;
local target_embedding_dim = 128;
local max_decoding_steps = 80;
local beam_size = 5;
local target_decoder_layers = 1;
local gamma = 2.0;
local scheduled_sampling_ratio = 0.0;

{
  "dataset_reader": {
    "type": "peter_reader",
    "binarizator": binarizator,
    "augmentator": augmentator,
    "lazy": true
  },
  "validation_dataset_reader": {
    "type": "peter_reader",
    "binarizator": binarizator,
    "augmentator": null,
    "lazy": false
  },
  "train_data_path": std.extVar("TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("VALID_DATA_PATH"),
  "vocabulary": COMMON["vocabulary"],
  "model": {
    "type": "generative_img2sentence",
    "max_decoding_steps": max_decoding_steps,
    "attention": {"type": "additive", "vector_dim": encoder_dim, "matrix_dim": encoder_dim},
    "target_embedding_dim": target_embedding_dim,
    "beam_size": beam_size,
    "scheduled_sampling_ratio": scheduled_sampling_ratio,
    "target_decoder_layers": target_decoder_layers,
    "gamma": gamma
  },
  "data_loader": COMMON['data_loader'],
  "trainer": {
    // -loss
    "validation_metric": "-loss",
    "num_epochs": 500,
    "patience": 50,
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    },
    "epoch_callbacks": ["wandb", "track_epoch_callback"],
    "cuda_device": 0
  }
}