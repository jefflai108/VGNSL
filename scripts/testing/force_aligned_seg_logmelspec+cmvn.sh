#!/bin/bash 

basename=$1
datadir=data/SpokenCOCO
expdir=exp/spokencoco/force_aligned_seg_logmelspec+cmvn_${basename}
i=0
while [ $i -ne 20 ]; do  
    if [ -f ${expdir}/${i}.pth.tar ]; then
        #echo evaluating ${i}.pth.tar
        python src/test.py --data_path ${datadir}/Freda-formatting/ --candidate ${expdir}/${i}.pth.tar --vocab_path ${datadir}/SpokenCOCO_vocab.pkl --basename ${basename}
    fi 
    i=$(($i+1))
done

