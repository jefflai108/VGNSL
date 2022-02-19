#!/bin/bash 

basename=$1
embed_size=$2
lr=$3
davenet_embed_type=$4
feature=logmelspec; feature_dim=40
datadir=data/SpokenCOCO
expdir=exp/spokencoco/force_aligned_scratch-${davenet_embed_type}_${feature}_embed${embed_size}_lr${lr}_${basename}

i=0
while [ $i -ne 20 ]; do  
    if [ -f ${expdir}/${i}.pth.tar ]; then
        #echo evaluating ${i}.pth.tar
        python src/test.py --data_path ${datadir}/Freda-formatting/ --candidate ${expdir}/${i}.pth.tar --vocab_path ${datadir}/SpokenCOCO_vocab-threshold1.pkl --basename ${basename} 
    fi 
    i=$(($i+1))
done
exit 0
