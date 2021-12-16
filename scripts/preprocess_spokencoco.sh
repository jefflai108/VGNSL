#!/bin/bash 

python data/preprocess_spokencoco.py -i data/SpokenCOCO/SpokenCOCO_val_wparse.json -d data/SpokenCOCO
python data/preprocess_spokencoco.py -i data/SpokenCOCO/SpokenCOCO_train_wparse.json -d data/SpokenCOCO
