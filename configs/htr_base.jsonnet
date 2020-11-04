{
    "dataset_reader": {
        "preprocessor": {
            "type": "compose",
            "preprocessors": [
                "basic_rotator",
                "basic_resizer",
                "null_binarizer",
            ]
        },
        "augmentator": "flor",
    },
    "model": {
        "type": "flor",
        "beam_size": 50,
    },
    "training": {
        "num_epochs": 300,
        "batch_size": 8,
        "learning_rate": 0.001,
        "patience": 20,
        "lr_patience": 15
    },
//    "postprocessor": {
//        "type": "compose",
//        "modules": [
//            {
//                "type": "lm",
//                "path": "presets/lm",
//            },
//            {
//                "type": "seq2seq",
//                "path": "presets/seq2seq"
//            },
//            {
//                "type": "regex"
//            }
//        ]
//    }
}