#!/bin/bash

python src/main.py train \
    --train-path "data/02-21.10way.clean" \
    --dev-path "data/22.auto.clean" \
    --use-chars-lstm --use-encoder --num-layers 8 \
    --batch-size 250 --learning-rate 0.0008 \
    --model-path-base models/English_charlstm
