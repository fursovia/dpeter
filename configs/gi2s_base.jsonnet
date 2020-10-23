local COMMON = import 'common/basic.jsonnet';

// do not change image size
local image_size = [1024, 128];

local binarizator = "null";
local augmentator = "google";

local encoder_dim = 288;
local target_embedding_dim = 64;
local max_decoding_steps = 80;
local beam_size = 3;

{
  "dataset_reader": {
    "type": "peter_reader",
    "image_size": image_size,
    "binarizator": binarizator,
    "augmentator": augmentator,
    "shuffle": true,
    "lazy": true
  },
  "validation_dataset_reader": {
    "type": "peter_reader",
    "image_size": image_size,
    "binarizator": binarizator,
    "augmentator": null,
    "shuffle": false,
    "lazy": false
  },
  "train_data_path": std.extVar("TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("VALID_DATA_PATH"),
  "model": {
    "type": "generative_img2sentence",
    "max_decoding_steps": max_decoding_steps,
    "attention": {"type": "additive", "vector_dim": target_embedding_dim, "matrix_dim": encoder_dim},
    "target_embedding_dim": target_embedding_dim,
    "beam_size": beam_size
  },
  "data_loader": COMMON['data_loader'],
  "trainer": {
    // -loss
    "validation_metric": "-cer",
    "num_epochs": 300,
    "patience": 20,
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    },
    "epoch_callbacks": ["wandb"],
    "cuda_device": 0
  }
}