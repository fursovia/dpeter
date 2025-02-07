local COMMON = import 'common/basic.jsonnet';

local binarizator = "simple";
local augmentator = {"type": "rotation", "degree": 3};
//local augmentator = {"type": "perspective_rotation", "degree": 3, "distortion_scale": 0.2, "p": 0.5, "interpolation": 3};

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