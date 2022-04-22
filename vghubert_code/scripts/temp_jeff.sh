model=$1 # 26
segment_layer=$2
snapshot=$3 # either 'best', or some number between 0 and 85 (inclusive)
k=$4 # 4096
threshold=$5 # 0.9
segment_method=$6
dataset=${7} # (lowercase) spokencoco, timit
vad=${8} # python, matlab, no
insertThreshold=${9}

# exp_dir: disc-81 or disc-82
# feature_layer 0, .., 11
    # concate vanilla hubert2
# snapshot: best (for now). 0, .., 85 (based on MBR selection)

# naming convention based on these 3 variables

python simplified_get_seg_feats.py \
--exp_dir /data/sls/scratch/clai24/syntax/VGNSL-feature/Jason_model_weights/disc-81 \ 
--feature_layer 9 \
--snapshot best
