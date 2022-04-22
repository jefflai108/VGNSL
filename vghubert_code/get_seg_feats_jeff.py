import argparse
from email.policy import default
import torch
import torch.nn as nn
import os
import pickle
import sys
import progressbar
import json
import math
import soundfile as sf
import tqdm
import torch.nn.functional as F
import time
import numpy as np
from torch.nn.parallel.data_parallel import DataParallel
from models import w2v2_model_jeff
import transformers
import tqdm
import numpy as np
import h5py
from itertools import groupby
from collections import defaultdict
from operator import itemgetter


def extract_feats(boundaries, feats, spf):
    seg_feats = [[] for j in range(len(feats))]
    for s, e in boundaries:
        for j in range(len(feats)):
            seg_feats[j].append(feats[j][int(s/spf):max(int(e/spf), int(s/spf)+1)])
    return seg_feats

def cls_attn_seg_extract_feats(feats, cls_attn_weights, threshold, spf, no_cls, vad, insert_threshold):
    # return a list of features that are segmented by cls attn weights
    threshold_value = torch.quantile(cls_attn_weights, threshold, dim=-1, keepdim=True) # [n_h, T]
    cls_attn_weights_sum = cls_attn_weights.sum(0)
    important_idx = torch.where((cls_attn_weights >= threshold_value).float().sum(0) > 0)[0].cpu().numpy()
    boundaries = []
    boundaries_all = []
    boundaries_ex1 = []
    for k, g in groupby(enumerate(important_idx), lambda ix : ix[0] - ix[1]):
        seg = list(map(itemgetter(1), g))
        t_s, t_e = seg[0], min(seg[-1]+1, cls_attn_weights.shape[-1])
        if len(seg) > 1:
            boundaries_all.append([t_s, t_e])
            boundaries_ex1.append([t_s, t_e])
        else:
            boundaries_all.append([t_s, t_e])
    
    if len(boundaries_ex1) == 0:
        boundaries = boundaries_all
    else:
        boundaries = boundaries_ex1
    # boundaries = boundaries_all
    if vad is None:
        vad = [[0.0, (cls_attn_weights.shape[-1]+1)*spf*100]]
    seg_feats = [[] for _ in range(len(feats))] # len(feats) is number of layers
    locations = []
    boundaries_in_sec = []
    for t_s, t_e in boundaries:
        for gt_b in vad:
            if (t_s*spf >= (gt_b[0]/100. - 0.02)) and (t_e*spf <= (gt_b[1]/100. + 0.02)):
                locations.append(spf*(t_s+t_e)/2.) # in seconds
                boundaries_in_sec.append([t_s*spf, t_e*spf]) # in seconds
                for j in range(len(feats)):
                    seg_feats[j].append(feats[j][t_s:t_e].cpu())
                break

    # Refinement based on VAD and insertion
    # delete all segments that completely fall into non-voiced region
    # first assign each boundaries to it's VAD region, and then draw word boundaries within that region, make sure the boundaries of the VAD is also word boundaries
    vad2bou = {}
    vad2loc = {}
    vad2feat = {}
    for gt_b in vad:
        vad2bou[f"{gt_b}"] = []
        vad2loc[f"{gt_b}"] = []
        vad2feat[f"{gt_b}"] = [[] for _ in range(len(feats))]
    for i in range(len(locations)):
        for gt_b in vad:
            if (locations[i] >= (gt_b[0]/100. - 0.02)) and (locations[i] <= (gt_b[1]/100. + 0.02)):
                vad2bou[f"{gt_b}"].append(boundaries_in_sec[i])
                vad2loc[f"{gt_b}"].append(locations[i])
                for  j in range(len(feats)):
                    vad2feat[f"{gt_b}"][j].append(seg_feats[j][i])
                break
    for gt_b in vad:
        if len(vad2bou[f"{gt_b}"]) == 0: # in case some vad region doesn't have any attn segments
            added_s, added_e = gt_b[0]/100., min(gt_b[0]/100.+ 0.05, gt_b[1]/100.)
            f_s, f_e = int(added_s / spf), max(int(added_e / spf), int(added_s / spf)+1)
            vad2bou[f"{gt_b}"].append([added_s, added_e])
            vad2loc[f"{gt_b}"].append((added_s+added_e)/2.)
            for  j in range(len(feats)):
                vad2feat[f"{gt_b}"][j].append(feats[j][f_s:f_e].cpu())
    # insert a segment in the middle when the gap between two adjacent segments are lower than threshold
    # also make sure the segment is in the voiced region
    
    interval = insert_threshold/2.
    for gt_b in vad:
        cur_boundaries_in_sec = vad2bou[f"{gt_b}"]
        for i in range(len(cur_boundaries_in_sec)):
            if i == 0:
                right_b = cur_boundaries_in_sec[i][0]
                left_b = gt_b[0]/100.
            elif i == len(cur_boundaries_in_sec) - 1:
                right_b = gt_b[1]/100.
                left_b = cur_boundaries_in_sec[i][1]
            else:
                right_b = cur_boundaries_in_sec[i+1][0]
                left_b = cur_boundaries_in_sec[i][1]

            gap = right_b - left_b
            if gap > insert_threshold:
                num_insert = int(gap/interval) - 1 # if two intervals can be put in the gap, then insert 1 seg to separate them, if 3 intervals can be put in the gap, then insert 2 seg to separate them...
                for insert_start in range(1, num_insert+1):
                    s_in_sec = left_b + insert_start * interval 
                    s_frame = max(int(left_b/spf), int(s_in_sec / spf))
                    e_frame = s_frame + 2
                    e_in_sec =  min(right_b, e_frame * spf)
                    vad2bou[f"{gt_b}"].append([s_in_sec, e_in_sec])
                    vad2loc[f"{gt_b}"].append((s_in_sec+e_in_sec)/2.)
                    for  j in range(len(feats)):
                        vad2feat[f"{gt_b}"][j].append(feats[j][s_frame:e_frame].cpu())
        cur_locations, sorted_ind = torch.sort(torch.tensor(vad2loc[f"{gt_b}"]))
        vad2loc[f"{gt_b}"] = cur_locations.tolist()
        vad2bou[f"{gt_b}"] = np.array(vad2bou[f"{gt_b}"])[sorted_ind].tolist()
        if not isinstance(vad2bou[f"{gt_b}"][0], list):
            vad2bou[f"{gt_b}"] = [vad2bou[f"{gt_b}"]]
        for  j in range(len(feats)):
            temp = vad2feat[f"{gt_b}"][j]
            vad2feat[f"{gt_b}"][j] = [temp[si] for si in sorted_ind]

    word_boundaries = []
    for i, gt_b in enumerate(vad):
        word_boundaries_line = [gt_b[0]/100.] # the first is vad boundary
        temp_boundaries = vad2bou[f"{gt_b}"]
        for left, right in zip(temp_boundaries[:-1], temp_boundaries[1:]):
            word_boundaries_line.append((left[1]+right[0])/2.)
        word_boundaries_line.append(gt_b[1]/100.)
        for i in range(len(word_boundaries_line)-1):
            word_boundaries.append([word_boundaries_line[i], word_boundaries_line[i+1]])
    seg_feats, locations, boundaries_in_sec = [[] for _ in range(len(feats))], [], []
    for gt_b in vad:
        for j in range(len(feats)):
            seg_feats[j] += vad2feat[f"{gt_b}"][j]
        locations += vad2loc[f"{gt_b}"]
        boundaries_in_sec += vad2bou[f"{gt_b}"]
    if len(seg_feats) == 0:
        seg_feats = [[feats[j].mean(0).cpu().unsqueeze(0)] for j in range(len(feats))]
        locations = [(cls_attn_weights.shape[-1])*spf/2.]
        boundaries_in_sec = [[0.0, (cls_attn_weights.shape[-1])*spf]]
        word_boundaries = [[0.0, (cls_attn_weights.shape[-1])*spf]]

    # print(len(seg_feats))
    # print(len(seg_feats[0]))
    # print(seg_feats[0][0].shape)
    # raise
    assert len(word_boundaries) == len(seg_feats[0]), f"seg_feats {len(seg_feats[0])}, locations {len(locations)}, boundaries_in_sec {len(boundaries_in_sec)}, word_boundaries {len(word_boundaries)}"
    return {"seg_feats": seg_feats, "locations": locations, "boundaries": boundaries_in_sec, "word_boundaries": word_boundaries}





print("I am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--exp_dir", type=str, help="the root where the model (e.g. full path to disc-81) is stored")
parser.add_argument("--dataset", type=str, default='spokencoco')
parser.add_argument("--seg_fn", type=str, default='no', help="path to the segmentation (i.e. boundaries) file, if not provided, use do segmentation and feature extraction on the fly")
parser.add_argument("--vad", type=str, choices=['python', 'matlab', 'no'], default='no')
parser.add_argument("--insert_threshold", type=float, default=10000.0, help="if the gap between two attention segments are above the threshold, we insert a two frame segment in the middle")
parser.add_argument("--snapshot", type=str, default='best', help='which model snapshot to use, best means best_boundle.pth, can also pass number x, say 24, then will take snapshot_24.pth')
parser.add_argument("--data_json", type=str, default="/data1/scratch/coco_pyp/SpokenCOCO/SpokenCOCO_val_unrolled_karpathy_with_alignments.json")
parser.add_argument("--audio_base_path", type=str, default="/data1/scratch/coco_pyp/SpokenCOCO")
parser.add_argument("--max_iter", type=int, default=200)
parser.add_argument("--num_heads", type=int, default=12)
parser.add_argument("--save_root", type=str, default="/data2/scratch/pyp/discovery/word_unit_discovery/", help="the root where the extracted boundaries and features are stored")
parser.add_argument("--threshold", type=float, default=0.90, help="attention threshold for segmentation")
parser.add_argument("--segment_layer", type=int, default=9, help='the layer from which we take cls attn to do segmentation (layer number is 0-based), if seg_fn is not no. then this is ignored and we use boundaries in seg_fn to extract features')
parser.add_argument("--feature_layer", nargs="+", type=int, help='which layers we want to extract feature from, can pass in a sequence of numbers like 2 9, which will be processed as list [2,9] (layer number is 0-based)')
parser.add_argument("--segment_method", type=str, choices=['clsAttn', 'forceAlign', 'uniform'], default=None, help="only clsAttn is avaliable")
args = parser.parse_args()

save_root = os.path.join(args.save_root, args.exp_dir.split("/")[-1])
feats_type = args.dataset  + "_" + str(args.threshold) + "_" + args.segment_method + "_" + str(args.segment_layer) + "_" + str(args.feature_layer) + "_" + "vad" + args.vad + "_" + "insertThreshold" + str(args.insert_threshold) + "_" + "snapshot" + args.snapshot

save_root = os.path.join(save_root, feats_type)
print("data save at: ", save_root)
os.makedirs(save_root, exist_ok=True)
print(args)
if not os.path.isdir(args.exp_dir):
    raise RuntimeError(f"{args.exp_dir} does not exist!!")

########################## setup model ##########################
with open(os.path.join(args.exp_dir, "args.pkl"), "rb") as f:
    model_args = pickle.load(f)
model = w2v2_model_jeff.Wav2Vec2Model_cls(model_args)
if "best" in args.snapshot:
    bundle = torch.load(os.path.join(args.exp_dir, "best_bundle.pth"))
else:
    snapshot = int(args.snapshot)
    bundle = torch.load(os.path.join(args.exp_dir, f"snapshot_{snapshot}.pth"))

if "dual_encoder" in bundle:
    model.carefully_load_state_dict(bundle['dual_encoder'], load_all=True)
elif "audio_encoder" in bundle:
    model.carefully_load_state_dict(bundle['audio_encoder'], load_all=True)
else:
    model.carefully_load_state_dict(bundle['model'], load_all=True)

model.eval()
model = model.cuda()
########################## setup model ##########################


data_start_time = time.time()

with open(args.data_json, "r") as f:
    data_json = json.load(f)['data']

if args.vad != "no":
    vad_json_fn = args.data_json.split(".")[0] + "_vad_" + args.vad + ".json"
    with open(vad_json_fn, "r") as f:
        vad_json = json.load(f)
else:
    vad_json = None

locF_temp = []
j = 0
# total_data = []
data_dict = {}
missing_ali = 0
level2 = False
if args.seg_fn != "no":
    assert os.path.isfile(args.seg_fn)
    with open(args.seg_fn, "rb") as f:
        segment_data = pickle.load(f)
for item in tqdm.tqdm(data_json):
    if args.dataset == "spokencoco":
        wav_fn = item['caption']['wav']
        key = os.path.join(args.audio_base_path, item['caption']['wav'])
    elif args.dataset == "timit":
        wav_fn = item['wav']
        key = item['wav']
    audio, sr = sf.read(key, dtype = 'float32')
    assert sr == 16000
    with torch.no_grad():
        # forward pass
        w2v2_out = model(torch.from_numpy(audio).unsqueeze(0).cuda(), padding_mask=None, mask=False, need_attention_weights=True, segment_layer=args.segment_layer, feature_layer=args.feature_layer, level2=level2)
    
    if args.segment_method == "clsAttn": # use cls attn for segmentation
        if not (model_args.use_audio_cls_token and model_args.cls_coarse_matching_weight > 0.):
            no_cls = True
        else:
            no_cls = False
        # this is the representations
        feats = [item.squeeze(0)[1:] for item in w2v2_out['layer_features']] # [1, T+1, D] -> [T, D]
        spf = len(audio)/sr/feats[0].shape[-2]
        attn_weights = w2v2_out['attn_weights'].squeeze(0) # [1, num_heads, tgt_len, src_len] -> [num_heads, tgt_len, src_len]
        if no_cls:
            cls_attn_weights = (attn_weights.sum(1) - attn_weights[:,range(attn_weights.shape[2]),range(attn_weights.shape[2])]).squeeze().cpu()
        else:
            cls_attn_weights = attn_weights[:, 0, 1:] # [num_heads, tgt_len, src_len] -> [n_h, T]
        if args.seg_fn == 'no':
            out = cls_attn_seg_extract_feats(feats, cls_attn_weights, args.threshold, spf, no_cls, None if vad_json is None else vad_json[wav_fn], args.insert_threshold)
        else:
            out = extract_feats(segment_data[wav_fn], feats) # only output seg_feats, of the shape [args.feature_layer, L, D] but note that it's a list of list of tensor

    elif args.segment_method == "forceAlign":
        raise NotImplementedError
        
    elif args.segment_method == "uniform":
        raise NotImplementedError
        


with open(os.path.join(save_root, 'data_dict.pkl'), "wb") as f:
    pickle.dump(data_dict, f)
print(f"save pickle data at {os.path.join(save_root, 'data_dict.pkl')}")


