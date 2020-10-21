local COMMON = import 'common/basic.jsonnet';

local image_size = [1024, 128];
local binarizator = "simple";
local augmentator = {"type": "rotation", "degree": 4};
// local augmentator = {"type": "perspective_rotation", "degree": 4, "distortion_scale": 0.25, "p": 0.7, "interpolation": 2};

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