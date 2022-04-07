import os.path as osp
import json 
import pickle
import argparse

import numpy as np 

def run(orig_word_list_pth, output_word_list_pth):
    orig_word_list = np.load(orig_word_list_pth, allow_pickle=True)[0]

    uniform_word_list = {}
    for i, sentence in orig_word_list.items():
        total_duration = sentence[-1][-1]
        num_of_words = len(sentence)
        duration_per_word = total_duration / num_of_words
        mod_sentence = [(word[0], k * duration_per_word, (k+1) * duration_per_word) for (k, word) in enumerate(sentence)]

        uniform_word_list[i] = mod_sentence

    np.save(output_word_list_pth, [uniform_word_list])

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig_word_seg_file', type=str)
    parser.add_argument('--output_word_seg_file', type=str)
    args = parser.parse_args()

    run(args.orig_word_seg_file, args.output_word_seg_file)
