# post-process Jason's word discovery features

import os.path as osp
import json 
import pickle
from tqdm import tqdm

import numpy as np 
import textgrid

from networkx_tutorial import run as l1_alignment

# 1. map "wav" --> location in "wavs-speaker"

# e.g. "wavs/train/428/mfbjxyldr6c1o-3ZPPDN2SLVWRQHXHKC1P7P9QLHI9EC_296759_180764.wav" 
# shoudl be "wavs-speaker/mfbjxyldr6c1o/mfbjxyldr6c1o-3ZPPDN2SLVWRQHXHKC1P7P9QLHI9EC_296759_180764.wav"
#
# 2. SpokenCOCO summary: SpokenCOCO_summary-83k-5k-5k.json
# find the corresponding files 
#
#
# 2.1 make sure the time stamps align (frame rate)
#
# 3. modify word_list (create a new one) with new word boundaries
# e.g. y = np.load('test_segment-hubert2_word_list-83k-5k-5k.npy', allow_pickle=True)
# >>> y[0][0]
#[('a', 33.5, 41.5), ('dirt', 41.5, 61.0), ('path', 61.0, 86.5), ('with', 86.5, 92.5), ('a', 92.5, 96.0), ('young', 96.0, 112.00000000000001), ('person', 112.00000000000001, 132.5), ('on', 132.5, 140.0), ('a', 140.0, 144.5), ('motor', 144.5, 159.5), ('bike', 159.5, 178.0), ('rests', 183.5, 214.5), ('to', 216.0, 221.49999999999997), ('the', 221.49999999999997, 226.5), ('foreground', 226.5, 260.0), ('of', 260.0, 265.0), ('a', 265.0, 268.5), ('verdant', 268.5, 292.0), ('area', 293.5, 314.5), ('with', 314.5, 324.0), ('a', 324.0, 327.5), ('bridge', 327.5, 357.0), ('and', 366.5, 373.0), ('a', 373.0, 377.5), ('background', 377.5, 410.50000000000006), ('of', 410.50000000000006, 416.5), ('cloudwreathed', 423.00000000000006, 469.00000000000006), ('mountains', 476.99999999999994, 509.49999999999994)]
##

# goal of the above: 
# first step: construct a dictionary {new_wav_id: y'[0][0]} 
# second step: min-weight matching of y[0][0] and y'[0][0]
# third step: output in Freda format by running same procedure of data/reformat_v4.py 

def _convert_default_wav_to_wavspeaker(main_dir, wav_pth): 
    wav_name = wav_pth.split('/')[-1]
    spk = wav_pth.split('/')[-1].split('-')[0]
    return osp.join(main_dir, 'wavs-speaker', spk, wav_name)

def create_iteration_list(data_summary_json, wav2wordseg, gt_word_lists, hubert_frame_stride=0.02): 
    idx = 0 
    over_cnt = 1
    for image_key, captions_list in tqdm(data_summary_json.items()):
        captions_list = _deduplicate(captions_list, image_key)
        for caption in captions_list: 
            wav_pth = caption[0]
            transcript_file = caption[1]
            alignment_file = caption[3]

            pred_word_segment = wav2wordseg[wav_pth]

            #gt_word_list = gt_word_lists[idx]
            #gt_word_list_seconds = [(x[0], x[1]*hubert_frame_stride, x[2]*hubert_frame_stride) for x in gt_word_list]

            gt_word_list = read_textgrid(alignment_file, transcript_file, pred_word_segment[-1][-1].item())
            print(pred_word_segment) # store this in the same format as gt_word_list
            print(gt_word_list)
            alignment = l1_alignment(gt_word_list, pred_word_segment)
            print(alignment) # store this in the same order as others

            #word_list, word_string = read_textgrid(alignment_file, transcript_file, nframes, frame_stride=self.logmelspec_frame_stride)

            #if word_segment[-1][-1].item() > gt_word_list_seconds[-1][-1]:
            #    print(over_cnt)
            #    print(word_segment)
            #    print(gt_word_list_seconds)
            #    over_cnt += 1

            idx += 1

def read_textgrid(textgrid_pth, text_pth, pred_last_timestamp, epsilon=0.001):
    # return [(0-th word, start frame, end frame), ..., (n-th word, start frame, end frame)]
    word_tgs = textgrid.TextGrid.fromFile(textgrid_pth)[0]

    word_list = []
    for word_tg in word_tgs: 
        word = word_tg.mark
        if word == '': # probably silence
            continue 
        word_obj = (word, word_tg.minTime, word_tg.maxTime)
        word_list.append(word_obj)
    assert word_tgs[-1].maxTime + epsilon >= pred_last_timestamp, print(word_tgs[-1], pred_last_timestamp)

    return word_list
 
def _deduplicate(captions_list, image_key):
    # ensure image:captions == 1:5
    if len(captions_list) > 5: 
        captions_list = captions_list[:5]
    while len(captions_list) < 5: # duplicate 
        print('duplicate %s captions' % image_key)
        captions_list.append(captions_list[-1])
    assert len(captions_list) == 5

    return captions_list

def read_jason_file(main_dir, ffile): 
    with open(ffile, 'rb') as f: 
        features = pickle.load(f)

    features = {_convert_default_wav_to_wavspeaker(main_dir, k):v['boundaries'] for k,v in features.items()}
    print(len(features.keys())) # 25020 for val, 567171 for train
    # to-do: merge train and val 
    #print(features)
    return features



def combine_h5_files(self, all_lab_embed_file, all_lab_len_file, seg_embed_h5_obj, word_list_pth, feature='logmelspec'):
    print('writing to h5 object')

    total_word_list_dict = {}
    idx = 0
    for lab_id in range(self.num_labs): # load lab files in-order
        lab_f = all_lab_embed_file[lab_id]
        word_list_dict = np.load(all_lab_len_file[lab_id], allow_pickle=True)[0]
        for tmp_idx, tmp_word_list in tqdm(word_list_dict.items()): 
            print(f'Processing lab {lab_id} and index {idx}')
            total_word_list_dict[idx] = tmp_word_list

            tmp_feat = lab_f[str(tmp_idx)][:]
            seg_embed_h5_obj.create_dataset(str(idx), data=tmp_feat)

            idx += 1

    np.save(word_list_pth, [total_word_list_dict])

if __name__ == '__main__': 
    main_dir = 'data/SpokenCOCO'
    new_feature = 'disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn'
    spoken_coco_json = 'data/SpokenCOCO/SpokenCOCO_summary-83k-5k-5k.json'
    word_list_file = 'val_segment-hubert2_word_list-83k-5k-5k.npy'
    split = 'val'

    gt_word_list = np.load(osp.join('data/SpokenCOCO/Freda-formatting/', word_list_file), allow_pickle=True)[0]
    
    with open(spoken_coco_json) as f: 
        coco_json = json.load(f) # for determining 'train', 'val', 'test'
    val_wav2wordseg = read_jason_file(main_dir, osp.join(main_dir, 'Jason_word_discovery', new_feature, 'val_data_dict.pkl'))
    #train_wav2wordseg = read_jason_file(main_dir, osp.join(main_dir, 'Jason_word_discovery', new_feature, 'train_data_dict.pkl'))
    #create_iteration_list(coco_json[split], train_wav2wordseg, gt_word_list)
    create_iteration_list(coco_json[split], val_wav2wordseg, gt_word_list)

