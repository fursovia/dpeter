#!/usr/bin/env bash

DATE=$(date +%H%M%S-%d%m)

allennlp train configs/copynet.jsonnet \
    -s ./logs/${DATE}-seq2seq \
    --include-package allennlp_models
