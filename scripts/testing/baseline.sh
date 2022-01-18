#!/bin/bash 

datadir=data/SpokenCOCO
expdir=exp/spokencoco/text_spokencoco2
i=0
while [ $i -ne 20 ]
do  
    echo ../output/${i}.pth.tar
    python src/test.py --candidate ${expdir}/${i}.pth.tar --vocab_path ${datadir}/SpokenCOCO_vocab.pkl --data_summary_json ${datadir}/SpokenCOCO_summary.json --image_hdf5 ${datadir}/SpokenCOCO_images.h5
    i=$(($i+1))
done
exit 0

    --data_path ${datadir} --data_summary_json ${datadir}/SpokenCOCO_summary.json --image_hdf5 ${datadir}/SpokenCOCO_images.h5 \

cd src
#python test.py --candidate ../output/model_best.pth.tar
#python test.py --candidate ../output/6.pth.tar
#python test.py --candidate ../output/7.pth.tar

i=0
while [ $i -ne 29 ]
do
    echo ../output/${i}.pth.tar
    python src/test.py --candidate ../output/${i}.pth.tar
    i=$(($i+1))
done