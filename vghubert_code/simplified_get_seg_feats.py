import argparse
import torch
import torch.nn as nn
import os
import pickle
import sys
import json
import math
import soundfile as sf
import tqdm
import torch.nn.functional as F
import time
import numpy as np
from models import w2v2_model_jeff
import tqdm
import numpy as np
import h5py
from collections import defaultdict

print("I am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--exp_dir", type=str, help="the root where the model (e.g. full path to disc-81) is stored")
parser.add_argument("--feature_layer", type=int, nargs="+")
parser.add_argument("--snapshot", type=str, default='best', help='which model snapshot to use, best means best_boundle.pth, can also pass number x, say 24, then will take snapshot_24.pth')
args = parser.parse_args()

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

audio = torch.randn(16000)
sr = 16000

#audio, sr = sf.read(key, dtype = 'float32')
assert sr == 16000
with torch.no_grad():
    # forward pass
    w2v2_out = model(audio.unsqueeze(0).cuda(), padding_mask=None, mask=False, need_attention_weights=True, segment_layer=-1, feature_layer=args.feature_layer)

    # this is the representations
    feats = [item.squeeze(0)[1:] for item in w2v2_out['layer_features']] # [1, T+1, D] -> [T, D]
    
    print(feats[0].shape) # torch.Size([49, 768])
