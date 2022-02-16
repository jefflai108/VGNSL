import nltk
import numpy as np
import os
import json
import h5py
from tqdm import tqdm
import random

import torch
import torch.utils.data as data

class PrecompDataset(data.Dataset):
    """ default + segment speech (.npy) """

    def __init__(self, data_path, data_split, vocab, basename,
                 load_img=True, img_dim=2048, utt_cmvn=False):
        self.vocab = vocab
        self.img_dim = img_dim

        # load captions
        self._load_captions(data_path, data_split, basename)

        # load speech features
        self._load_speech_feature(data_path, data_split, basename, utt_cmvn)
        
        # load image features
        self._load_img_feature(data_path, data_split, basename, load_img)

    def _load_captions(self, data_path, data_split, basename): 
        # captions
        self.captions = list()
        with open(os.path.join(data_path, f'{data_split}_caps-{basename}.txt'), 'r') as f:
            for line in f:
                self.captions.append(line.strip().lower().split())
            f.close()
        self.length = len(self.captions)

    def _load_speech_feature(self, data_path, data_split, basename, utt_cmvn): 
        self.doc_segment_spec = np.load(os.path.join(data_path, f'{data_split}_segment-logmelspec_embed-{basename}.npy')) # (50000, 50, 40)
        self.logmelspec_dim = self.doc_segment_spec[0].shape[-1]
        self.logmelspec_true_len = np.load(os.path.join(data_path, f'{data_split}_segment-logmelspec_len-{basename}.npy'))
        if utt_cmvn:
            print('apply utterance-level CMVN')
            self.doc_segment_spec = self._utt_cmvn()
        assert len(self.doc_segment_spec) == self.length

    def _load_img_feature(self, data_path, data_split, basename, load_img=True):
        # image features
        if load_img:
            self.images = np.load(os.path.join(data_path, f'{data_split}_ims-{basename}.npy'))
        else:
            self.images = np.zeros((self.length // 5, self.img_dim))

        # each image can have 1 caption or 5 captions
        if self.images.shape[0] != self.length:
            self.im_div = 5
            assert self.images.shape[0] * 5 == self.length
        else:
            self.im_div = 1

    def _utt_cmvn(self):
        """ utterance-level CMVN. Within each utterance, minus the mean vector and divide by its std.
        need to be careful of padding. Operation is for each utterance.
        """
        unpadded_doc_segment_spec = []
        norm_doc_segment_spec = np.zeros((self.doc_segment_spec.shape))
        for i, sentence_segment_spec in enumerate(self.doc_segment_spec):
            true_len = self.logmelspec_true_len[i]
            unpadded_sentence_segment_spec = sentence_segment_spec[:true_len, :]
            norm_doc_segment_spec[i, :true_len] = self._cmvn(unpadded_sentence_segment_spec)
        return norm_doc_segment_spec

    def _cmvn(self, x, doc_dim=0):
        return (x - x.mean(doc_dim, keepdims=True))/ (1e-10 + x.std(doc_dim, keepdims=True))

    def _get_image_item(self, index): 
        img_id = index  // self.im_div
        image = torch.tensor(self.images[img_id])
        return image, img_id

    def _get_caption_item(self, index): 
        caption = [self.vocab(token)
                   for token in ['<start>'] + self.captions[index] + ['<end>']]
        return torch.tensor(caption)

    def _get_speech_item(self, index): 
        # account for start and end tokens
        dummy_segment_embed = np.zeros((1, self.logmelspec_dim)) 
        audio = np.concatenate((dummy_segment_embed, self.doc_segment_spec[index], dummy_segment_embed), axis=0)
        true_audio_len = self.logmelspec_true_len[index] + 2
        return torch.tensor(audio), true_audio_len

    def __getitem__(self, index):
        # get image 
        image, img_id = self._get_image_item(index)
        
        # get caption
        caption = self._get_caption_item(index)
        
        # get speech
        audio, true_audio_len = self._get_speech_item(index)
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

class H5PrecompDataset(PrecompDataset):
    """ default + whole speech (.hdf5) 
        re-use functions from PrecompDataset.
    """

    def __init__(self, data_path, data_split, vocab, basename,
                 load_img=True, img_dim=2048, feature='logmelspec', utt_cmvn=False):
        self.data_split = data_split 
        self.vocab = vocab
        self.img_dim = img_dim

        # load captions
        self._load_captions(data_path, data_split, basename)
    
        # load speech features
        self._load_speech_feature(data_path, data_split, basename, feature, utt_cmvn)

        # load image features
        self._load_img_feature(data_path, data_split, basename, load_img)

    def _load_speech_feature(self, data_path, data_split, basename, feature='logmelspec', utt_cmvn=False): 
        # whole hubert
        self.feature_embed_obj = h5py.File(os.path.join(data_path, f'{data_split}_segment-{feature}_embed-{basename}.hdf5'), 'r')
        self.feature_wordlist = np.load(os.path.join(data_path, f'{data_split}_segment-{feature}_word_list-{basename}.npy'), allow_pickle=True)[0]
        #print(self.feature_embed_obj[str(22)][:].shape)
        #print(self.feature_wordlist[22])
        
        self.feature_dim = self.feature_embed_obj[str(0)][:].shape[-1]
        #print(self._slice_speech_feature(self.feature_embed_obj[str(22)][:], self.feature_wordlist[22])[0].shape)

        #if utt_cmvn:
        #    print('apply utterance-level CMVN')
        #    self.doc_segment_spec = self.utt_cmvn()
        #assert len(self.doc_segment_spec) == self.length
        assert len(self.feature_wordlist.keys()) == self.length

    def _slice_speech_feature(self, feat, word_list, max_segment_len=50):
        # return (n-th word, word segment # frames, feature-dim), where 1st dim is padded to longest segment frame for an given utterance

        assert len(feat) >= round(word_list[-1][-1]), print(word_list, len(feat))
        word2len = [round(z)-round(y) for (_,y,z) in word_list]
        #sliced_feat = np.zeros((len(word_list), max(word2len), self.feature_dim)) # e.g. (9, 33, 768)
        sliced_feat = np.zeros((len(word_list), max_segment_len, self.feature_dim)) # limit the segment-dimension length

        for i, (word, start_frame, end_frame) in enumerate(word_list):
            start_frame, end_frame = round(start_frame), round(end_frame)
            if end_frame - start_frame > max_segment_len: 
                segment_len = max_segment_len
                start_frame = random.randint(start_frame, end_frame - max_segment_len)
                #print('cropping speech segments: original %f, orig start frame is %f, new start frame is %f' % (end_frame - start_frame, orig_start_frame, start_frame))
                end_frame   = start_frame + max_segment_len  
            else: 
                segment_len = word2len[i]
            sliced_feat[i, :segment_len] = feat[start_frame:end_frame, :]

        return sliced_feat, len(word_list)
 
    def _get_speech_item(self, index): 
        ## audio (slicing on-the-fly)
        whole_feature_embed = self.feature_embed_obj[str(index)][:]
        sliced_feature_embed, len_word_list = self._slice_speech_feature(whole_feature_embed, self.feature_wordlist[index])
        # account for start and end tokens (dummy_pad)
        dummy_segment_embed = np.zeros((1, sliced_feature_embed.shape[1], self.feature_dim))
        audio = np.concatenate((dummy_segment_embed, sliced_feature_embed, dummy_segment_embed), axis=0)
        true_audio_len = len_word_list + 2
        segment_len = sliced_feature_embed.shape[1]
        return torch.tensor(audio), true_audio_len, segment_len

    def __getitem__(self, index):
        # get image 
        image, img_id = self._get_image_item(index)
        
        # get caption
        caption = self._get_caption_item(index)
        
        # get speech
        audio, true_audio_len, segment_len = self._get_speech_item(index)
        assert true_audio_len == len(caption)

        return image, caption, audio, true_audio_len, segment_len, index, img_id

def h5_collate_fn(data):
    """ build mini-batch tensors from a list of (image, caption) tuples """
    # sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    zipped_data = list(zip(*data))
    images, captions, audios, true_audio_lens, audio_segment_lens, ids, img_ids = zipped_data
    images = torch.stack(images, 0)
    max_sentence_len = len(captions[0])
    targets = torch.zeros(len(captions), max_sentence_len).long()
    lengths = [len(cap) for cap in captions] # --> ensure this match with true_audio_lens
    for i, cap in enumerate(captions):
        end = len(cap)
        targets[i, :end] = cap[:end]

    # padding in the sentence-level and segment-level
    max_audio_segment_len = max(list(audio_segment_lens))
    feature_dim = audios[0].shape[-1]
    target_audios = torch.zeros(len(captions), max_sentence_len, max_audio_segment_len, feature_dim).float()
    for i, audio in enumerate(audios):
        true_sentence_len = true_audio_lens[i]
        true_audio_segment_len = audio_segment_lens[i]
        target_audios[i, :true_sentence_len, :true_audio_segment_len] = audio
    sentence_level_speech_masks = torch.tensor(true_audio_lens)
    audio_masks = torch.where(target_audios==0, -100000, 0) # mask==-inf indicates padding 
    audio_masks = audio_masks[:, :, :, 0].squeeze(-1) # feature-dim is not indicative 
    #print(target_audios.shape, audio_masks.shape) # torch.Size([256, 27, 71, 768]) torch.Size([256, 27, 71])
    assert sentence_level_speech_masks.tolist() == lengths # ensure we can match segment embed to words

    return images, targets, target_audios, audio_masks, lengths, ids

class H5DiscretePrecompDataset(PrecompDataset):
    """ default + whole speech (.hdf5) with Discrete IDs as input
        re-use functions from PrecompDataset.
    """

    def __init__(self, data_path, data_split, vocab, basename,
                 load_img=True, img_dim=2048, feature='logmelspec', utt_cmvn=False, 
                 discretized_phone=False, discretized_word=False, km_clusters=0):
        self.data_split = data_split 
        self.vocab = vocab
        self.img_dim = img_dim
        self.discretized_phone = discretized_phone
        self.discretized_word  = discretized_word
        self.km_clusters = km_clusters

        # load captions
        self._load_captions(data_path, data_split, basename)
    
        # load speech features
        self._load_speech_feature(data_path, data_split, basename, feature, utt_cmvn, km_clusters)
        self.pad_discrete_token_id = km_clusters+2
        self.start_discrete_token_id, self.end_discrete_token_id = km_clusters, km_clusters+1

        # load image features
        self._load_img_feature(data_path, data_split, basename, load_img)

    def _load_speech_feature(self, data_path, data_split, basename, feature='logmelspec', utt_cmvn=False, km_clusters=0): 
        # whole hubert
        self.feature_embed_obj = h5py.File(os.path.join(data_path, f'{data_split}_segment-{feature}_embed-km{km_clusters}-{basename}.hdf5'), 'r')
        self.feature_wordlist = np.load(os.path.join(data_path, f'{data_split}_segment-{feature}_word_list-{basename}.npy'), allow_pickle=True)[0]
        #print(self.feature_embed_obj[str(22)][:])
        #print(self.feature_wordlist[22])
        
        assert len(self.feature_wordlist.keys()) == self.length

    def _slice_speech_feature(self, feat, word_list, max_segment_len=50):
        # return (n-th word, word segment # frames), where 1st dim is padded to longest segment frame for an given utterance

        assert len(feat) >= round(word_list[-1][-1]), print(word_list, len(feat))
        word2len = [round(z)-round(y) for (_,y,z) in word_list]
        sliced_feat = self.pad_discrete_token_id * np.ones((len(word_list), max_segment_len)) # limit the segment-dimension length

        for i, (word, start_frame, end_frame) in enumerate(word_list):
            start_frame, end_frame = round(start_frame), round(end_frame)
            if end_frame - start_frame > max_segment_len: 
                segment_len = max_segment_len
                start_frame = random.randint(start_frame, end_frame - max_segment_len)
                #print('cropping speech segments: original %f, orig start frame is %f, new start frame is %f' % (end_frame - start_frame, orig_start_frame, start_frame))
                end_frame   = start_frame + max_segment_len  
            else: 
                segment_len = word2len[i]
            deduplicate_segment = self._lossy_run_length_encoding(feat[start_frame:end_frame])
            sliced_feat[i, :len(deduplicate_segment)] = deduplicate_segment

        return sliced_feat, len(word_list)

    def _get_word_level_speech_item(self, index): 
        """ different from `discretized_phone`, the self._lossy_run_length_encoding() is applied *within* a given word segment 
            e.g. this is valid: <boundary> 1 2 1 <boundary> 1 3 2
        """
        whole_feature_embed = self.feature_embed_obj[str(index)][:]
        sliced_feature_embed, len_word_list = self._slice_speech_feature(whole_feature_embed, self.feature_wordlist[index])
        
        # account for start and end tokens (dummy_pad)
        start_segment_embed = self.pad_discrete_token_id * np.ones((1, sliced_feature_embed.shape[1]))
        end_segment_embed = self.pad_discrete_token_id * np.ones((1, sliced_feature_embed.shape[1]))
        start_segment_embed[0, 0] = self.start_discrete_token_id
        end_segment_embed[0, 0] = self.end_discrete_token_id
        audio = np.concatenate((start_segment_embed, sliced_feature_embed, end_segment_embed), axis=0)
        true_audio_len = len_word_list + 2

        return torch.tensor(audio), true_audio_len

    def _get_phone_level_speech_item(self, index): 
        """ remove duplicate consequtive ids in the given discrete_id sequence
        """
        whole_feature_embed = self.feature_embed_obj[str(index)][:]
        whole_feature_embed = np.insert(whole_feature_embed, 0, self.start_discrete_token_id)
        whole_feature_embed = np.append(whole_feature_embed, self.end_discrete_token_id)
        deduplicated_whole_feature_embed = torch.tensor(self._lossy_run_length_encoding(whole_feature_embed))

        return deduplicated_whole_feature_embed, deduplicated_whole_feature_embed.shape[-1]

    def _lossy_run_length_encoding(self, id_seq_list): 
        return [id for i, id in enumerate(id_seq_list) if i == 0 or id != id_seq_list[i-1]]

    def __getitem__(self, index):
        # get image 
        image, img_id = self._get_image_item(index)
        
        # get caption
        caption = self._get_caption_item(index)
 
        # get speech
        if self.discretized_word: 
            audio, true_audio_len = self._get_word_level_speech_item(index)
            assert true_audio_len == len(caption)
        elif self.discretized_phone: 
            audio, true_audio_len = self._get_phone_level_speech_item(index)
        
        return image, caption, audio, true_audio_len, index, img_id, self.km_clusters

def h5_discrete_collate_fn(data):
    """ build mini-batch tensors from a list of (image, caption) tuples """
    # sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    zipped_data = list(zip(*data))
    images, captions, audios, true_audio_lens, ids, img_ids, km_clusters = zipped_data
    images = torch.stack(images, 0)
    max_sentence_len = len(captions[0])
    targets = torch.zeros(len(captions), max_sentence_len).long()
    lengths = [len(cap) for cap in captions] # --> ensure this match with true_audio_lens
    for i, cap in enumerate(captions):
        end = len(cap)
        targets[i, :end] = cap[:end]

    pad_discrete_token_id = km_clusters[0] + 2
    if len(audios[0].shape) == 2: # discretized_word == True 
        # padding in the sentence-level and segment-level
        max_audio_segment_len = 50
        target_audios = pad_discrete_token_id * torch.ones(len(captions), max_sentence_len, max_audio_segment_len).long()
        for i, audio in enumerate(audios):
            true_sentence_len = true_audio_lens[i]
            target_audios[i, :true_sentence_len] = audio
        sentence_level_speech_masks = torch.tensor(true_audio_lens)
        audio_masks = torch.where(target_audios==pad_discrete_token_id, -100000, 0) # mask==-inf indicates padding 
        #print(target_audios.shape, audio_masks.shape) # torch.Size([128, 22, 50]) torch.Size([128, 22, 50])
        #print(target_audios[22, 0:3], audio_masks[22, 0:3]) # torch.Size([128, 502]) torch.Size([128, 502])
        assert sentence_level_speech_masks.tolist() == lengths # ensure we can match segment embed to words
    elif len(audios[0].shape) == 1: # discretized_phone == True
        # padding in the sentence-level 
        max_sentence_len = max(true_audio_lens)
        target_audios = pad_discrete_token_id * torch.ones(len(captions), max_sentence_len).long()
        for i, audio in enumerate(audios):
            true_sentence_len = true_audio_lens[i]
            target_audios[i, :true_sentence_len] = audio
        sentence_level_speech_masks = torch.tensor(true_audio_lens)
        audio_masks = torch.where(target_audios==pad_discrete_token_id, -100000, 0) # mask==-inf indicates padding 
        #print(target_audios.shape, audio_masks.shape) # torch.Size([128, 502]) torch.Size([128, 502])

    return images, targets, target_audios, audio_masks, lengths, ids

def get_precomp_loader(data_path, data_split, vocab, basename, 
                       batch_size=128, shuffle=True, num_workers=2, load_img=True, img_dim=2048, 
                       feature='logmelspec', utt_cmvn=False, speech_hdf5=False, 
                       discretized_phone=False, discretized_word=False, km_clusters=0):
    if speech_hdf5: # whole utterance, support for logmelspec and hubert 
        if discretized_phone or discretized_word: 
            dset = H5DiscretePrecompDataset(data_path, data_split, vocab, basename, load_img, img_dim, feature, utt_cmvn, 
                                            discretized_phone, discretized_word, km_clusters)
            data_loader = torch.utils.data.DataLoader(
                dataset=dset, batch_size=batch_size, shuffle=shuffle,
                pin_memory=True,
                collate_fn=h5_discrete_collate_fn
            )
        else:
            dset = H5PrecompDataset(data_path, data_split, vocab, basename, load_img, img_dim, feature, utt_cmvn)
            data_loader = torch.utils.data.DataLoader(
                dataset=dset, batch_size=batch_size, shuffle=shuffle,
                pin_memory=True,
                collate_fn=h5_collate_fn
            )
    else: # averaged over segments, support for logmelspec 
        dset = PrecompDataset(data_path, data_split, vocab, basename, load_img, img_dim, utt_cmvn)
        data_loader = torch.utils.data.DataLoader(
            dataset=dset, batch_size=batch_size, shuffle=shuffle,
            pin_memory=True,
            collate_fn=collate_fn
        )

    return data_loader


def get_train_loaders(data_path, vocab, basename, batch_size, workers, feature='logmelspec', utt_cmvn=False, speech_hdf5=False, 
                     discretized_phone=False, discretized_word=False, km_clusters=0):
    
    assert discretized_phone & discretized_word == False

    train_loader = get_precomp_loader(
        data_path, 'train', vocab, basename, batch_size, True, workers, feature=feature, utt_cmvn=utt_cmvn, speech_hdf5=speech_hdf5, 
        discretized_phone=discretized_phone, discretized_word=discretized_word, km_clusters=km_clusters
    )
    val_loader = get_precomp_loader(
        data_path, 'val', vocab, basename, batch_size, False, workers, feature=feature, utt_cmvn=utt_cmvn, speech_hdf5=speech_hdf5, 
        discretized_phone=discretized_phone, discretized_word=discretized_word, km_clusters=km_clusters
    )
    return train_loader, val_loader


def get_eval_loader(data_path, split_name, vocab, basename, batch_size, workers,
                    feature='logmelspec', speech_hdf5=False, load_img=False, img_dim=2048, utt_cmvn=False, 
                    discretized_phone=False, discretized_word=False, km_clusters=0):

    assert discretized_phone & discretized_word == False
    
    eval_loader = get_precomp_loader(
        data_path, split_name, vocab, basename, batch_size, False, num_workers=0, feature=feature, 
        speech_hdf5=speech_hdf5, load_img=load_img, img_dim=img_dim, utt_cmvn=utt_cmvn, 
        discretized_phone=discretized_phone, discretized_word=discretized_word, km_clusters=km_clusters
    )
    return eval_loader
