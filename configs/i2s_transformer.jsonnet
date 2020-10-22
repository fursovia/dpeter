local COMMON = import 'common/basic.jsonnet';

// do not change image size
local image_size = [1024, 128];

local binarizator = "simple";
local augmentator = {"type": "rotation", "degree": 3};
//local augmentator = {"type": "perspective_rotation", "degree": 3, "distortion_scale": 0.2, "p": 0.5, "interpolation": 3};

local tr_input_size = 128;

local length_classifier_path = "presets/lc.model.tar.gz";
local embedding_dim = 128;
local attention_vector_dim = 128;

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
    "type": "img2sentence",
    "seq2seq_encoder": {
      "type": "pytorch_transformer",
      "input_dim": tr_input_size,
      "num_layers": 4,
      "num_attention_heads": 4,
      "feedforward_hidden_dim": 32,
      "positional_embedding_size": 512,
      "positional_encoding": "embedding",
      "dropout_prob": 0.05,
      "activation": "relu"
    },
    "length_classifier": {
      "type": "from_archive",
      "archive_file": length_classifier_path
    },
    "input_dim": tr_input_size,
    "emb_dim": embedding_dim,
    "att_dim": attention_vector_dim,
    "regularizer": {
      "regexes": [
        [".*", {
          "type": "l2",
          "alpha": 1e-07
        }]
      ]
    }
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
    "cuda_device": 0
  }
}