#!/bin/bash

#cd src
datadir=data/SpokenCOCO
python data/data_reformat.py \
    --data_summary_json ${datadir}/SpokenCOCO_summary.json --image_hdf5 ${datadir}/SpokenCOCO_images.h5 \
    -o data/SpokenCOCO/reformat/
