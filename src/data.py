import nltk
import numpy as np
import os
import json 
import h5py
from tqdm import tqdm

import torch
import torch.utils.data as data

class PrecompDataset(data.Dataset):
    """ add add """

    def __init__(self, data_path, data_split, vocab, basename,
                 load_img=True, img_dim=2048):
        self.vocab = vocab

        # captions
        self.captions = list()
        with open(os.path.join(data_path, f'{data_split}_caps-{basename}.txt'), 'r') as f:
            for line in f:
                self.captions.append(line.strip().lower().split())
            f.close()
        self.length = len(self.captions)

        # segment logmelspec 
        self.doc_segment_spec = np.load(os.path.join(data_path, f'{data_split}_segment-logmelspec_embed-{basename}.npy')) # (50000, 50, 40)
        self.logmelspec_dim = self.doc_segment_spec[0].shape[-1]
        self.logmelspec_true_len = np.load(os.path.join(data_path, f'{data_split}_segment-logmelspec_len-{basename}.npy'))
        assert len(self.doc_segment_spec) == self.length 

        # image features
        if load_img:
            self.images = np.load(os.path.join(data_path, f'{data_split}_ims-{basename}.npy'))
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

        # audio: account for start and end tokens 
        dummy_segment_embed = np.zeros((1, self.logmelspec_dim))
        audio = np.concatenate((dummy_segment_embed, self.doc_segment_spec[index], dummy_segment_embed), axis=0)
        audio = torch.tensor(audio)
        true_audio_len = self.logmelspec_true_len[index] + 2
        assert true_audio_len == len(caption)

        return image, caption, audio, true_audio_len, index, img_id

    def __len__(self):
        return self.length

def collate_fn(data):
    """ build mini-batch tensors from a list of (image, caption) tuples """
    # sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    zipped_data = list(zip(*data))
    images, captions, audios, true_audio_lens, ids, img_ids = zipped_data
    images = torch.stack(images, 0)
    max_sentence_len = len(captions[0])
    targets = torch.zeros(len(captions), max_sentence_len).long()
    lengths = [len(cap) for cap in captions] # --> ensure this match with true_audio_lens
    for i, cap in enumerate(captions):
        end = len(cap)
        targets[i, :end] = cap[:end]

    target_audios = torch.stack(list(audios), dim=0).float() # torch.Size([B, 52, 40])
    target_audios = target_audios[:, :max_sentence_len, :] # truncate target_audios to match the sentence_len of targets
    audio_masks = torch.tensor(true_audio_lens)
    assert audio_masks.tolist() == lengths # ensure we can match segment embed to words 
    
    return images, targets, target_audios, audio_masks, lengths, ids

def get_precomp_loader(data_path, data_split, vocab, basename, batch_size=128,
                       shuffle=True, num_workers=2, load_img=True, 
                       img_dim=2048):
    dset = PrecompDataset(data_path, data_split, vocab, basename, load_img, img_dim)
    data_loader = torch.utils.data.DataLoader(
        dataset=dset, batch_size=batch_size, shuffle=shuffle,
        pin_memory=True, 
        collate_fn=collate_fn
    )
    return data_loader


def get_train_loaders(data_path, vocab, basename, batch_size, workers):
    train_loader = get_precomp_loader(
        data_path, 'train', vocab, basename, batch_size, True, workers
    )
    val_loader = get_precomp_loader(
        data_path, 'val', vocab, basename, batch_size, False, workers
    )
    return train_loader, val_loader


def get_eval_loader(data_path, split_name, vocab, basename, batch_size, workers, 
                    load_img=False, img_dim=2048):
    eval_loader = get_precomp_loader(
        data_path, split_name, vocab, basename, batch_size, False, workers, 
        load_img=load_img, img_dim=img_dim
    )
    return eval_loader
