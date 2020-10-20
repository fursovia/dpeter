local COMMON = import 'common/basic.jsonnet';

local image_size = [1024, 128];
local augmentator = null;

{
  "dataset_reader": {
    "type": "peter_reader",
    "image_size": image_size,
    "augmentator": augmentator,
    "lazy": true
  },
  "validation_dataset_reader": {
    "type": "peter_reader",
    "image_size": image_size,
    "augmentator": null,
    "lazy": false
  },
  "train_data_path": std.extVar("TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("VALID_DATA_PATH"),
  "model": {
    "type": "length_classifier",
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
  "trainer": COMMON['trainer']
}