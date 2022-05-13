#!/bin/bash 

basename=$1
embed_size=$2
lr=$3
feature=hubert; feature_dim=768
feature=hubert2; feature_dim=768
feature=hubert4; feature_dim=768
feature=hubert6; feature_dim=768
feature=hubert8; feature_dim=768
feature=hubert10; feature_dim=768
feature=$4
datadir=data/SpokenCOCO
spokencoco_expdir=exp/spokencoco/force_aligned_whole_${feature}_embed${embed_size}_lr${lr}_${basename}
echo $spokencoco_expdir

i=0
while [ $i -ne 20 ]; do  
    if [ -f ${spokencoco_expdir}/${i}.pth.tar ]; then
        #echo evaluating ${i}.pth.tar
        python src/test.py --data_path data/LJspeech/Freda-formatting/ --candidate ${spokencoco_expdir}/${i}.pth.tar --vocab_path ${datadir}/SpokenCOCO_vocab-threshold1.pkl \
                           --basename LJdev --data_split dev --ljspeech 
    fi 
    i=$(($i+1))
done
exit 0
