#!/usr/bin/env bash

DATE=$(date +%H%M%S-%d%m)

allennlp train configs/seq2seq.jsonnet \
    -s ./logs/${DATE}-seq2seq \
    --include-package allennlp_models
