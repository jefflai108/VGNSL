import nltk
import numpy as np
import os.path as op
import json 
import h5py
from tqdm import tqdm
import argparse

import torch
import torch.utils.data as data

from utils import compute_spectrogram, read_textgrid, slice_spectrogram

class SummaryJsonReader(object):
    def __init__(self, data_summary_json, image_hdf5, img_embed_dim=2048):

        self.image_h5 = h5py.File(image_hdf5, 'r')
        self.data_summary_json = data_summary_json
        self.img_embed_dim = img_embed_dim

        print('pre-store ordered image_key to ensure the storing order is consistent')
        self.image_key_list = []
        for image_key, captions_list in tqdm(data_summary_json.items()):
            self.image_key_list.append(image_key)

    def write_image_to_file(self, output_pth): 
        print('writing image_embed to %s' % output_pth)
        image_embed = np.zeros((len(self.image_key_list), self.img_embed_dim))
        for idx, image_key in enumerate(self.image_key_list): 
            image_embed[idx] = self.image_h5[image_key][:]
        print(image_embed.shape)
        np.save(output_pth, image_embed)

    def write_text_or_tree_to_file(self, tree_pth, transcript_pth): 
        print('writing tree to %s and transcript to %s' % (tree_pth, transcript_pth))
        tree_f = open(tree_pth, 'w')
        transcript_f = open(transcript_pth, 'w')
        for image_key, captions_list in tqdm(self.data_summary_json.items()): 
            captions_list = self.__deduplicate__(captions_list, image_key)
            for captions in captions_list: 
                transcript_file = captions[1]
                tree_file = captions[2]

                transcript_f.write('%s\n' % self.__readfile__(transcript_file))
                tree_f.write('%s\n' % self.__readfile__(tree_file))
    
    def __deduplicate__(self, captions_list, image_key):
        # ensure image:captions == 1:5
        if len(captions_list) > 5: 
            captions_list = captions_list[:5]
        while len(captions_list) < 5: # duplicate 
            print('duplicate %s captions' % image_key)
            captions_list.append(captions_list[-1])
        assert len(captions_list) == 5

        return captions_list

    @staticmethod
    def __readfile__(fpath): 
        with open(fpath, 'r') as f: 
            string = f.readline()
        return string 

    def write_utterance_to_file(self, seg_embed_pth, true_len_pth):
        print('writing logmelspec segment embeddings to %s and true utterance length to %s' % (seg_embed_pth, true_len_pth)) 

        padding_len = 50
        logmelspec_dim = 40

        doc_segment_spec = np.zeros((len(self.image_key_list)*5, padding_len, logmelspec_dim))
        num_of_words_list = []
        idx = 0
        for image_key, captions_list in tqdm(self.data_summary_json.items()): 
            captions_list = self.__deduplicate__(captions_list, image_key)
            for captions in captions_list: 
                transcript_file = captions[1]
                wav_file = captions[0]
                alignment_file = captions[3]

                logspec, nframes = compute_spectrogram(wav_file) # (40, 530)
                word_list, word_string = read_textgrid(alignment_file, transcript_file, nframes)
                sentence_segment_spec, num_of_words = slice_spectrogram(logspec, word_list, padding_len, True) # (50, 40), 10
                #print(sentence_segment_spec.shape, num_of_words)
                doc_segment_spec[idx] = sentence_segment_spec
                num_of_words_list.append(num_of_words)
                idx += 1

        num_of_words_list = np.array(num_of_words_list)
        np.save(true_len_pth, num_of_words_list)
        np.save(seg_embed_pth, doc_segment_spec)
           
def main(args):
    
    basename = '-'.join(args.data_summary_json.split('-')[1:]).split('.')[0]
    print('processing %s' % basename)

    with open(args.data_summary_json, 'r') as f:
        data_summary = json.load(f)
    train_json = data_summary['train']
    val_json   = data_summary['val']
    test_json  = data_summary['test'] 

    print('loading val_loader')
    val_writer = SummaryJsonReader(val_json, args.image_hdf5)
    #val_writer.write_image_to_file(op.join(args.output_dir, 'val_ims-' + basename + '.npy'))
    #val_writer.write_text_or_tree_to_file( \
    #    op.join(args.output_dir, 'val_ground-truth-' + basename + '.txt'), op.join(args.output_dir, 'val_caps-' + basename + '.txt'))
    val_writer.write_utterance_to_file( \
        op.join(args.output_dir, 'val_segment-logmelspec_embed-' + basename + '.npy'), op.join(args.output_dir, 'val_segment-logmelspec_len-' + basename + '.npy'))

    print('loading test_loader')
    test_writer = SummaryJsonReader(test_json, args.image_hdf5)
    #test_writer.write_image_to_file(op.join(args.output_dir, 'test_ims-' + basename + '.npy'))
    #test_writer.write_text_or_tree_to_file( \
    #    op.join(args.output_dir, 'test_ground-truth-' + basename + '.txt'), op.join(args.output_dir, 'test_caps-' + basename + '.txt'))
    test_writer.write_utterance_to_file( \
        op.join(args.output_dir, 'test_segment-logmelspec_embed-' + basename + '.npy'), op.join(args.output_dir, 'test_segment-logmelspec_len-' + basename + '.npy'))

    print('loading train_loader')
    train_writer = SummaryJsonReader(train_json, args.image_hdf5)
    #train_writer.write_image_to_file(op.join(args.output_dir, 'train_ims-' + basename + '.npy'))
    #train_writer.write_text_or_tree_to_file( \
    #    op.join(args.output_dir, 'train_ground-truth-' + basename + '.txt'), op.join(args.output_dir, 'train_caps-' + basename + '.txt'))
    train_writer.write_utterance_to_file( \
        op.join(args.output_dir, 'train_segment-logmelspec_embed-' + basename + '.npy'), op.join(args.output_dir, 'train_segment-logmelspec_len-' + basename + '.npy'))

if __name__ == '__main__': 
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-summary-json', '-j', type=str)
    parser.add_argument('--image-hdf5', '-i', type=str)
    parser.add_argument('--output-dir', '-o', type=str)
    args = parser.parse_args()

    main(args)
