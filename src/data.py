import nltk
import numpy as np
import os
import json 
import h5py
from tqdm import tqdm

import torch
import torch.utils.data as data

class OriginalPrecompDataset(data.Dataset):
    """ load precomputed captions and image features """

    def __init__(self, data_path, data_split, vocab, 
                 load_img=True, img_dim=2048):
        self.vocab = vocab

        # captions
        self.captions = list()
        with open(os.path.join(data_path, f'{data_split}_caps.txt'), 'r') as f:
            for line in f:
                self.captions.append(line.strip().lower().split())
            f.close()
        self.length = len(self.captions)

        # image features
        if load_img:
            self.images = np.load(os.path.join(data_path, f'{data_split}_ims.npy'))
        else:
            self.images = np.zeros((self.length // 5, img_dim))
        
        # each image can have 1 caption or 5 captions 
        if self.images.shape[0] != self.length:
            self.im_div = 5
            assert self.images.shape[0] * 5 == self.length
        else:
            self.im_div = 1

    def __getitem__(self, index):
        # image
        img_id = index  // self.im_div
        image = torch.tensor(self.images[img_id])
        # caption
        caption = [self.vocab(token) 
                   for token in ['<start>'] + self.captions[index] + ['<end>']]
        caption = torch.tensor(caption)
        return image, caption, index, img_id

    def __len__(self):
        return self.length

class PrecompDataset(data.Dataset):
    """ load precomputed captions and image features """

    def __init__(self, data_summary_json, vocab, image_hdf5, 
                 load_img=True, img_dim=2048):
        self.vocab = vocab

        print('load', image_hdf5)
        image_h5 = h5py.File(image_hdf5, 'r')

        print('read & extract data_summary_json')
        self.key2embed = {}
        self.transcript_list = []
        transcript_cnt = 0
        self.cnt2key = {}
        for image_key, captions_list in tqdm(data_summary_json.items()): 
            if load_img:
                image_embed = image_h5[image_key][:] # numpy array 
            else:
                image_embed = np.zeros(img_dim,)
            self.key2embed[image_key] = image_embed

            for captions in captions_list: 
                wav_file = captions[0]
                transcript_file = captions[1]
                tree_file = captions[2]
                alignment_file = captions[3]
                
                self.transcript_list.append(self.__readfile__(transcript_file))
                self.cnt2key[transcript_cnt] = image_key
                transcript_cnt += 1 # since dataset access is based on this counter, set # of workers = 0 

            if transcript_cnt >= 30: 
                break 
    
        print('length of transcription is %d' % len(self.transcript_list)) 
        self.length = len(self.transcript_list)

    @staticmethod
    def __readfile__(fpath): 
        with open(fpath, 'r') as f: 
            string = f.readline()
        return string 

    def __getitem__(self, index):
        # transcript 
        transcript_tmp = self.transcript_list[index].split()
        caption = [self.vocab(token) 
                   for token in ['<start>'] + transcript_tmp + ['<end>']]
        caption = torch.tensor(caption)

        # image
        image_key = self.cnt2key[index]
        image = torch.tensor(self.key2embed[image_key])

        return image, caption, index, image_key

    def __len__(self):
        return self.length

def collate_fn(data):
    """ build mini-batch tensors from a list of (image, caption) tuples """
    # sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    zipped_data = list(zip(*data))
    images, captions, ids, keys = zipped_data
    images = torch.stack(images, 0)
    targets = torch.zeros(len(captions), len(captions[0])).long()
    lengths = [len(cap) for cap in captions]
    for i, cap in enumerate(captions):
        end = len(cap)
        targets[i, :end] = cap[:end]
    return images, targets, lengths, ids, keys

def get_precomp_loader(data_summary_json, vocab, image_hdf5, batch_size=128, 
                       shuffle=True, num_workers=2, load_img=True, img_dim=2048):
    dset = PrecompDataset(data_summary_json, vocab, 
                          image_hdf5, load_img, img_dim)
    data_loader = torch.utils.data.DataLoader(
        dataset=dset, batch_size=batch_size, shuffle=shuffle,
        pin_memory=True, 
        collate_fn=collate_fn
    )

    return data_loader

def get_train_loaders(data_path, vocab, data_summary_json, image_hdf5, batch_size, workers):
    with open(data_summary_json, 'r') as f:
        data_summary = json.load(f)
    train_json = data_summary['train']
    val_json = data_summary['val']

    print('loading train_loader')
    train_loader = get_precomp_loader(
        train_json, vocab, image_hdf5, batch_size, True, workers
    )
    print('loading val_loader')
    val_loader = get_precomp_loader(
        val_json, vocab, image_hdf5, 1, False, workers # have to set batch_size=1 during val/test, since collate_fn sorts data by length (messes up the order)
    )
    return train_loader, val_loader

def get_eval_loader(data_path, split_name, vocab, batch_size, workers, 
                    load_img=False, img_dim=2048):
    with open(data_summary_json, 'r') as f:
        data_summary = json.load(f)

    test_json = data_summary['test']

    eval_loader = get_precomp_loader(
        data_path, split_name, vocab, 1, False, workers, 
        load_img=load_img, img_dim=img_dim
    )
    return eval_loader
