#!/bin/bash 

basename=$1
embed_size=$2
lr=$3
rerun=$4
feature=hubert2; feature_dim=768
feature=$5
datadir=data/SpokenCOCO
expdir=exp/spokencoco/force_aligned_whole_${feature}_embed${embed_size}_lr${lr}_run${rerun}_${basename}
i=0
while [ $i -ne 20 ]; do  
    if [ -f ${expdir}/${i}.pth.tar ]; then
        #echo evaluating ${i}.pth.tar
        python src/test.py --data_path ${datadir}/Freda-formatting/ --candidate ${expdir}/${i}.pth.tar --vocab_path ${datadir}/SpokenCOCO_vocab-threshold1.pkl --basename ${basename} 
    fi 
    i=$(($i+1))
done
exit 0
