import json 
import argparse 
from tqdm import tqdm
import os
import os.path as osp
from os.path import exists
import copy 
import regex

import numpy as np
import textgrid
from nltk import Tree

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

    def write_phn_tree_to_file(self, phn_tree_pth, word_tree_pth, new_word_tree_pth, 
                               phn_caps_pth): 
        print('writing phone-level tree to %s and word-level tree to %s' % (phn_tree_pth, new_word_tree_pth))
        print('writing phone-level captions to %s' % (phn_caps_pth))
        phn_tree_f = open(phn_tree_pth, 'w')
        new_word_tree_f = open(new_word_tree_pth, 'w')
        phn_cap_f  = open(phn_caps_pth, 'w')
        gt_word_level_trees = self._read_text_file(word_tree_pth)
        tree_cnt = 0 
        _mismatch_cnt = 0 
        for image_key in tqdm(self.utt_key_list):
            captions_list = self.data_summary_json[image_key]
            captions_list = self.__deduplicate__(captions_list, image_key)
            for captions in captions_list: 
                transcript_file = captions[1]
                word_tree_file  = captions[2]
                textgrid_file   = captions[3]
                
                word_level_tree = self._read_text_file(word_tree_file)[0]
                gt_word_level_tree = gt_word_level_trees[tree_cnt] 
                assert word_level_tree == gt_word_level_tree # double-ensure *ordering* is correct
                tree_cnt += 1

                # ensure the underlying sentence from tree == ground-truth sentence 
                words_from_gt_tree = self._retrieve_sentence_from_tree(gt_word_level_tree)
                word2phn, gt_text, phn_string = self._construct_word2phn_mapping(textgrid_file, transcript_file)
                if words_from_gt_tree != gt_text: 
                    # if there's sentence mismatch, add a placeholder string. This will be handy during training/testing tree eval. 
                    phn_level_tree = 'MISMATCH'
                    gt_word_level_tree = 'MISMATCH'
                    _mismatch_cnt += 1
                else:
                    # if no sentence mismatch, generate phn-level tree
                    phn_level_tree = self._convert_tree_via_word2phn(word_level_tree, word2phn, tree_cnt, viz=False)
                    phns_from_gt_tree = self._retrieve_sentence_from_tree(phn_level_tree)
                    assert phns_from_gt_tree == phn_string
            
                phn_tree_f.write('%s\n' % phn_level_tree)
                new_word_tree_f.write('%s\n' % gt_word_level_tree)
                phn_cap_f.write('%s\n' % phn_string)
        print('number of MISMATCH is %d' % _mismatch_cnt)
   
    def _retrieve_sentence_from_tree(self, tree): 
        return ' '.join([_x for _x in tree.split() if _x.isalnum()])
    
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
                curr_word = word_list[word_cnt][0] 
                # use (word, word_cnt) as key, because a word can be mapped to different sets of phones 
                # depending on punctuations. e.g. a --> EY1 or a --> AH0. 
                word2phn[(curr_word, word_cnt)] = tmp_phn_list 
                tmp_phn_list = []
                word_cnt += 1
            phn_string.append(phn)

        word_string = ' '.join(word_string)
        phn_string  = ' '.join(phn_string)
        with open(text_pth, 'r') as f: 
            gt_text = f.readline()
        assert gt_text == word_string, print(f'{word_string}\n{gt_text}')
        
        return word2phn, gt_text, phn_string 

    def _convert_tree_via_word2phn(self, word_level_tree, word2phn, tree_cnt, viz=False): 
        word_level_tree_list = word_level_tree.split()
        phn_level_tree_list = []
        word_cnt = 0
        for idx, word_token in enumerate(word_level_tree_list): 
            if word_token in ['(', ')']: 
                phn_level_tree_list.append(word_token)
                continue 
            phns = word2phn[(word_token, word_cnt)]
            word_cnt += 1
            if len(phns) == 1: 
                phn_level_tree_list.append(phns[0])
            else: 
                phn_level_tree_list.append('(')
                phn_level_tree_list.extend(phns)
                phn_level_tree_list.append(')')
        
        phn_level_tree = ' '.join(phn_level_tree_list)
        if viz and tree_cnt < 100: # visualize top 100 samples 
            print('\n ground-truth word-level tree')
            viz_tree(word_level_tree)
            print('\n ground-truth phone-level tree via force alignment')
            viz_tree(phn_level_tree)

        return phn_level_tree

    def write_phn_list_to_numpy(self, phn_list_pth, word_list_pth, feature='logmelspec'):
        preprocessed_word_lists = np.load(word_list_pth, allow_pickle=True)[0]
        phn_list_dict = {}
        cnt = 0
        for image_key in tqdm(self.utt_key_list):
            captions_list = self.data_summary_json[image_key]
            captions_list = self.__deduplicate__(captions_list, image_key)
            for captions in captions_list: 
                textgrid_file = captions[3]
 
                if feature == 'logmelspec':
                    frame_stride = self.logmelspec_frame_stride 
                elif feature in ['hubert', 'hubert_large', 'content_vec_v07_11', 'content_vec_v12_05']:
                    frame_stride = self.hubert_frame_stride 
                max_frame = preprocessed_word_lists[cnt][-1][-1]
                phn_list = self._construct_phn_list(textgrid_file, max_frame, frame_stride)
                phn_list_dict[cnt] = phn_list
                cnt += 1

        np.save(phn_list_pth, [phn_list_dict])

    def _construct_phn_list(self, textgrid_pth, word_max_frame, frame_stride):
        # return [(0-th phn, start frame, end frame), ..., (n-th phn, start frame, end frame)]
        # 
        # note: logmelspec has frame_stride 10ms, while SSL models like hubert has 20ms 
        word_tgs, phn_tgs = textgrid.TextGrid.fromFile(textgrid_pth)

        phn_list = []
        for phn_tg in phn_tgs: 
            phn = phn_tg.mark
            if phn == '': # probably silence
                continue 
            # convert to frame-based (0.01s/0.02s stride)
            phn_obj = (phn, phn_tg.minTime/frame_stride, phn_tg.maxTime/frame_stride)
            phn_max_frame = phn_tg.maxTime/frame_stride
            phn_list.append(phn_obj)
        assert phn_max_frame == word_max_frame, print(phn_max_frame, word_max_frame) # word & phone end frame should match 

        return phn_list

def viz_tree(bare_tree):
    nt_tree = bare_tree.replace('(', '(NT').replace(' ', '  ')
    nt_tree = regex.sub(r' ([^ \(\)]+) ', r' (PT \1) ', nt_tree)
    nltk_tree = Tree.fromstring(nt_tree)
    nltk_tree.pretty_print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_summary_json', '-j', type=str)
    parser.add_argument('--data_dir', type=str, default='data/SpokenCOCO/Freda-formatting/')
    parser.add_argument('--data_split', type=str, choices=['test', 'val', 'train'])
    parser.add_argument('--feature', '-f', type=str, default='logmelspec', 
                        choices = ['logmelspec', 'hubert', 'hubert_large', 'content_vec_v07_11', 'content_vec_v12_05'])
    parser.add_argument('--layer_num', type=int, default=12)
    args = parser.parse_args()

    basename = '-'.join(args.data_summary_json.split('-')[1:]).split('.')[0]
    print('processing %s' % basename)

    with open(args.data_summary_json, 'r') as f:
        data_summary = json.load(f)[args.data_split]

    writer = PhoneTreeWriter(data_summary)
    # Step 1: convert word-level tree to phn-level based on force alignments
    writer.write_phn_tree_to_file(osp.join(args.data_dir, args.data_split + '_phn-level-ground-truth-' + basename + '.txt'), 
                                  osp.join(args.data_dir, args.data_split + '_ground-truth-' + basename + '.txt'), 
                                  osp.join(args.data_dir, args.data_split + '_word-level-ground-truth-' + basename + '.txt'), 
                                  osp.join(args.data_dir, args.data_split + '_phn_caps-' + basename + '.txt'))

    # Step 2: convert word_list to phn_list for logmelspec/hubert based on force alignments 
    #if args.feature == 'logmelspec' or (args.feature == 'hubert' and args.layer_num == 12): # naming convention 
    #    phn_list_pth  = osp.join(args.data_dir, f'{args.data_split}_segment-{args.feature}_phn_list-' + basename + '.npy')
    #    word_list_pth = osp.join(args.data_dir, f'{args.data_split}_segment-{args.feature}_word_list-' + basename + '.npy')
    #else:
    #    phn_list_pth  = osp.join(args.data_dir, f'{args.data_split}_segment-{args.feature}{args.layer_num}_phn_list-' + basename + '.npy')
    #    word_list_pth = osp.join(args.data_dir, f'{args.data_split}_segment-{args.feature}{args.layer_num}_word_list-' + basename + '.npy')
    #writer.write_phn_list_to_numpy(phn_list_pth, word_list_pth, args.feature)
