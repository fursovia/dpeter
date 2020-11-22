{
    "dataset_reader": {
        "preprocessor": {
            "type": "compose",
            "preprocessors": [
                "basic_rotator",
                "basic_resizer",
                "null_binarizer",
                "smart_flipper",
            ]
        },
        "augmentator": "flor",
    },
    "model": {
        "type": "flor",
        "beam_size": 50,
    },
    "training": {
        "num_epochs": 150,
        "batch_size": 8,
        "learning_rate": 0.001,
        "patience": 20,
        "lr_patience": 15
    },
    "postprocessor": {
        "type": "compose",
        "postprocessors": [
            {
                "type": "seq2seq",
                "archive_path": "presets/seq2seq.tar.gz",
                "beam_size": 10
            },
        ]
    }
}