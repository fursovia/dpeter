#!/usr/bin/env bash

DATE=$(date +%H%M%S-%d%m)

mkdir -p submits
zip -r submits/${DATE}_dpeter.zip dpeter/ presets/ tf_predict.py
zip -urj submits/${DATE}_dpeter.zip dpeter/commands/tf_predict.py presets/metadata.json