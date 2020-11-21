local target_namespace = "target_tokens";

{
    "dataset_reader": {
        "type": "copynet_seq2seq",
        "target_namespace": target_namespace,
        "source_tokenizer": "character",
        "target_tokenizer": "character",
    },
    "train_data_path": "data/seq2seq/train.tsv",
    "validation_data_path": "data/seq2seq/valid.tsv",
    "model": {
        "type": "copynet_seq2seq",
        "source_embedder": {
          "token_embedders": {
            "tokens": {
              "type": "embedding",
              "embedding_dim": 128,
              "vocab_namespace": "tokens"
            }
          }
        },
        "encoder": {
          "type": "lstm",
          "input_size": 128,
          "hidden_size": 256,
          "num_layers": 1,
          "dropout": 0.1,
          "bidirectional": false
        },
        "attention": {
            "type": "bilinear",
            "vector_dim": 256,
            "matrix_dim": 256,
        },
        "target_embedding_dim": 128,
        "beam_size": 1,
        "max_decoding_steps": 80,
    },
    "data_loader": {
        "shuffle": true,
        "batch_size": 512,
        "num_workers": 0,
        "batches_per_epoch": 1400,
        "pin_memory": true
    },
    "validation_data_loader": {
        "shuffle": false,
        "batch_size": 256,
        "num_workers": 0,
        "pin_memory": true
    },
    "distributed": {
//        "master_address": "127.0.0.1",
//        "master_port": 29502,
//        "num_nodes": 1,
        "cuda_devices": [
          0,
          1,
          3,
          4,
          5
        ]
  },
    "trainer": {
        "num_epochs": 300,
        "patience": 2,
//        "cuda_device": 0,
        "optimizer": {
          "type": "adam",
          "lr": 0.001
        }
    }
}