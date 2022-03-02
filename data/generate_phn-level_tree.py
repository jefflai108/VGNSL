import json 
import argparse 
from tqdm import tqdm
import os
import os.path as osp
from os.path import exists
import copy 

import numpy as np
import textgrid

class PhoneTreeWriter(object):
    def __init__(self, data_summary_json):

        self.data_summary_json = data_summary_json

        self.logmelspec_frame_stride = 0.01
        self.hubert_frame_stride = 0.02
                
        print('pre-store ordered image_key to ensure the storing order is consistent')
        self.image_key_list = []
        for image_key, captions_list in tqdm(data_summary_json.items()):
            self.image_key_list.append(image_key)
        self.utt_key_list = self.image_key_list

    def write_phn_tree_to_file(self, phn_tree_pth, word_tree_pth): 
        print('writing phone-level tree to %s' % (phn_tree_pth))
        phn_tree_f = open(phn_tree_pth, 'w')
        gt_word_level_trees = self._read_text_file(word_tree_pth)
        cnt = 0 
        for image_key in tqdm(self.utt_key_list):
            captions_list = self.data_summary_json[image_key]
            captions_list = self.__deduplicate__(captions_list, image_key)
            for captions in captions_list: 
                transcript_file = captions[1]
                word_tree_file  = captions[2]
                textgrid_file   = captions[3]
                
                word_level_tree = self._read_text_file(word_tree_file)[0]
                gt_word_level_tree = gt_word_level_trees[cnt] 
                assert word_level_tree == gt_word_level_tree # double-ensure ordering is correct
                cnt += 1

                word2phn = self._construct_word2phn_mapping(textgrid_file, transcript_file)
                phn_level_tree = self._convert_tree_via_word2phn(word_level_tree, word2phn)

                phn_tree_f.write('%s\n' % phn_level_tree)
    
    def __deduplicate__(self, captions_list, image_key):
        # ensure image:captions == 1:5
        if len(captions_list) > 5: 
            captions_list = captions_list[:5]
        while len(captions_list) < 5: # duplicate 
            print('duplicate %s captions' % image_key)
            captions_list.append(captions_list[-1])
        assert len(captions_list) == 5

        return captions_list

    def _read_text_file(self, f_pth): 
        with open(f_pth, 'r') as f: 
            lines = f.readlines()
        lines = [x.strip('\n') for x in lines]
        return lines

    def _construct_word2phn_mapping(self, textgrid_pth, text_pth):
        word_tgs, phn_tgs = textgrid.TextGrid.fromFile(textgrid_pth)

        word_list, word_string = [], []
        for word_tg in word_tgs: 
            word = word_tg.mark
            if word == '': # probably silence
                continue 
            word_obj = (word, word_tg.minTime, word_tg.maxTime)
            word_list.append(word_obj)
            word_string.append(word)

        word2phn, tmp_phn_list, phn_string = {}, [], []
        word_cnt = 0
        for phn_tg in phn_tgs: 
            phn = phn_tg.mark
            if phn == '': # probably silence
                continue 
            tmp_phn_list.append(phn)
            if phn_tg.maxTime == word_list[word_cnt][2]: # end-point detector 
                word2phn[word_list[word_cnt][0]] = tmp_phn_list 
                tmp_phn_list = []
                word_cnt += 1
            phn_string.append(phn)

        word_string = ' '.join(word_string)
        phn_string  = ' '.join(phn_string)
        with open(text_pth, 'r') as f: 
            gt_text = f.readline()
        assert gt_text == word_string, print(f'{word_string}\n{gt_text}')

        return word2phn

    def _convert_tree_via_word2phn(self, word_level_tree, word2phn): 
        word_level_tree_list = word_level_tree.split()
        phn_level_tree_list = []
        for idx, word_token in enumerate(word_level_tree_list): 
            if word_token in ['(', ')']: 
                phn_level_tree_list.append(word_token)
                continue 
            phns = word2phn[word_token]
            if len(phns) == 1: 
                phn_level_tree_list.append(phns[0])
            else: 
                phn_level_tree_list.append('(')
                phn_level_tree_list.extend(phns)
                phn_level_tree_list.append(')')
       
        return ' '.join(phn_level_tree_list)

    def _read_textgrid2(self, textgrid_pth, text_pth, nframes, frame_stride=0.01):
        # return [(0-th word, start frame, end frame), ..., (n-th word, start frame, end frame)]
        # 
        # note: logmelspec has frame_stride 10ms, while SSL models like hubert has 20ms 
        word_tgs = textgrid.TextGrid.fromFile(textgrid_pth)[0]

        word_list = []
        word_string = []
        for word_tg in word_tgs: 
            word = word_tg.mark
            if word == '': # probably silence
                continue 
            # convert to frame-based (0.01s/0.02s stride)
            word_obj = (word, word_tg.minTime/frame_stride, word_tg.maxTime/frame_stride)
            word_list.append(word_obj)
            word_string.append(word)
        assert word_tg.maxTime <= nframes

        word_string = ' '.join(word_string)
        with open(text_pth, 'r') as f: 
            gt_text = f.readline()
        assert gt_text == word_string, print(f'{word_string}\n{gt_text}')

        return word_list, word_string

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_summary_json', '-j', type=str)
    parser.add_argument('--data_dir', type=str, default='data/SpokenCOCO/Freda-formatting/')
    args = parser.parse_args()

    data_split = 'val'
    basename = '-'.join(args.data_summary_json.split('-')[1:]).split('.')[0]
    print('processing %s' % basename)

    with open(args.data_summary_json, 'r') as f:
        data_summary = json.load(f)[data_split]

    reader = PhoneTreeWriter(data_summary)
    reader.write_phn_tree_to_file(osp.join(args.data_dir, data_split + '_phn-level-ground-truth-' + basename + '.txt'), 
                                  osp.join(args.data_dir, data_split + '_ground-truth-' + basename + '.txt'))
