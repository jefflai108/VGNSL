#!/bin/bash

python src/main.py test \
    --test-path "data/23.auto.clean" \
    --no-predict-tags \
    --model-path models/English_charlstm_dev*.pt
