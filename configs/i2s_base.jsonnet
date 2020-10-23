local COMMON = import 'common/basic.jsonnet';

local binarizator = "null";
local augmentator = "null";

local gru_input_size = 128;
local gru_hidden_size = 256;
local gru_num_layers = 2;
local gru_dropout = 0.2;
local gru_bidirectionality = true;

local length_classifier_path = "presets/lc.model.tar.gz";
local embedding_dim = 128;
local attention_vector_dim = 128;

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
    "type": "img2sentence",
    "seq2seq_encoder": {
      "type": "gru",
      "input_size": gru_input_size,
      "hidden_size": gru_hidden_size,
      "num_layers": gru_num_layers,
      "dropout": gru_dropout,
      "bidirectional": gru_bidirectionality
    },
    "length_classifier": {
      "type": "from_archive",
      "archive_file": length_classifier_path
    },
    "input_dim": gru_input_size,
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
    "epoch_callbacks": ["wandb"],
    "cuda_device": 0
  }
}