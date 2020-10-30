#!/usr/bin/env bash

DATE=$(date +%H%M%S-%d%m)

mkdir -p submits
zip -r submits/${DATE}_dpeter.zip dpeter/ presets/ metadata.json tf_predict.py
