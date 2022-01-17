import nltk
import numpy as np
import os.path as op
import json 
import h5py
from tqdm import tqdm
import argparse

import torch
import torch.utils.data as data

def main(args):

    with open(args.data_summary_json, 'r') as f:
        data_summary = json.load(f)
    #train_json = data_summary['train']
    val_json = data_summary['val']
    test_json = data_summary['val']

    print('load ', args.image_hdf5)
    image_h5 = h5py.File(args.image_hdf5, 'r')

    write_to_file(train_json, image_h5, op.join(args.output_dir, 'train_ims.npy'), 
                  op.join(args.output_dir, 'train_caps.txt'), 
                  op.join(args.output_dir, 'train_ground-truth.txt')

def write_to_file(data_summary_json, image_h5, 
                  output_image_npy, transcript_fpath, tree_fpath, embed_dim=2048): 

    print('read & extract data_summary_json')
    image_embeds = np.zeros((len(data_summary_json.keys()), embed_dim))
    cnt = 0 
    transcript_f = open(transcript_fpath, 'w') 
    tree_f = open(tree_fpath, 'w') 
    for image_key, captions_list in tqdm(data_summary_json.items()):
        image_embeds[cnt] = image_h5[image_key][:] # numpy array
        cnt += 1
        for captions in captions_list:
            wav_file = captions[0]
            transcript_file = captions[1]
            tree_file = captions[2]
            alignment_file = captions[3]

            transcript_f.write(__readfile__(transcript_file))
            tree_f.write(__readfile__(tree_file))
    transcript_f.close()
    tree_f.close()
    
    print('image embedding shape', image_embeds.shape)
    np.save(output_image_npy, image_embeds)

def __readfile__(fpath): 
    with open(fpath, 'r') as f: 
        string = f.readline()
    return string + '\n'

if __name__ == '__main__': 
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-summary-json', '-j', type=str)
    parser.add_argument('--image-hdf5', '-h', type=str)
    parser.add_argument('--output-dir', '-o', type=str)
    args = parser.parse_args()

    split_id2path(args)
   
