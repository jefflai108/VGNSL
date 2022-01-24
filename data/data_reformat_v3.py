import nltk
import numpy as np
import os.path as op
import json 
import h5py
from tqdm import tqdm
import argparse

from utils import compute_spectrogram, read_textgrid, slice_spectrogram, hubert_feature_extraction

class SummaryJsonReader(object):
    def __init__(self, data_summary_json, image_hdf5, img_embed_dim=2048,
                 parallelize=False, num_labs=None, lab_id=None):

        self.image_h5 = h5py.File(image_hdf5, 'r')
        self.data_summary_json = data_summary_json
        self.img_embed_dim = img_embed_dim

        # for parallel extraction 
        self.num_labs = num_labs

        # for speech features
        self.padding_len = 50
        self.logmelspec_dim = 40
        self.hubert_dim = 768

        print('pre-store ordered image_key to ensure the storing order is consistent')
        self.image_key_list = []
        for image_key, captions_list in tqdm(data_summary_json.items()):
            self.image_key_list.append(image_key)

        if parallelize: 
            print('parallelize speech feature extraction')
            if lab_id >= num_labs: 
                print(f'invalid lab_id specification: {lab_id} exceeds {num_labs}')
                exit()
            start_idx = len(self.image_key_list) // num_labs * lab_id
            end_idx = len(self.image_key_list) // num_labs * (lab_id + 1)
            if lab_id == num_labs - 1: 
                end_idx = None
            self.utt_image_key_list = self.image_key_list[start_idx:end_idx]
        else: 
            self.utt_image_key_list = self.image_key_list[:]
        
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
        for image_key in tqdm(self.image_key_list):
            captions_list = self.data_summary_json[image_key]
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

    def write_utterance_to_file(self, seg_embed_pth, true_len_pth, feature='logmelspec'):
        print(f'writing {feature} segment embeddings to %s and true utterance length to %s' % (seg_embed_pth, true_len_pth)) 

        if feature == 'logmelspec':
            doc_segment_spec = np.zeros((len(self.utt_image_key_list)*5, self.padding_len, self.logmelspec_dim))
        elif feature == 'hubert': 
            doc_segment_spec = np.zeros((len(self.utt_image_key_list)*5, self.padding_len, self.hubert_dim))
        num_of_words_list = []
        idx = 0
        for image_key in tqdm(self.utt_image_key_list):
            captions_list = self.data_summary_json[image_key]
            captions_list = self.__deduplicate__(captions_list, image_key)
            for captions in captions_list: 
                transcript_file = captions[1]
                wav_file = captions[0]
                alignment_file = captions[3]
            
                if feature == 'logmelspec':
                    feat, nframes = compute_spectrogram(wav_file) # (40, 530)
                    word_list, word_string = read_textgrid(alignment_file, transcript_file, nframes, frame_stride=0.01)
                elif feature == 'hubert':
                    feat, nframes = hubert_feature_extraction(wav_file, layer=12) # (768, 264)
                    word_list, word_string = read_textgrid(alignment_file, transcript_file, nframes, frame_stride=0.02)
                
                sentence_segment_spec, num_of_words = slice_spectrogram(feat, word_list, \
                                                                        target_padded_length=self.padding_len, padding=True, \
                                                                        return_whole=False) # (50, 40), 10

                #print(feat.shape)
                #print(sentence_segment_spec.shape, num_of_words)
                doc_segment_spec[idx] = sentence_segment_spec
                num_of_words_list.append(num_of_words)
                idx += 1

        num_of_words_list = np.array(num_of_words_list)
        np.save(true_len_pth, num_of_words_list)
        np.save(seg_embed_pth, doc_segment_spec)

    def combine_lab_files(self, all_lab_embed_file, all_lab_len_file, seg_embed_pth, true_len_pth, feature='logmelspec'):
        print(f'writing {feature} segment embeddings to %s and true utterance length to %s' % (seg_embed_pth, true_len_pth)) 

        doc_segment_spec_list = []
        num_of_words_list = []
        for lab_id in range(self.num_labs): # load lab files in-order
            doc_segment_spec_list.append(np.load(all_lab_embed_file[lab_id]))
            num_of_words_list.extend(np.load(all_lab_len_file[lab_id]))
        doc_segment_spec = np.concatenate(doc_segment_spec_list, axis=0)
        num_of_words_list = np.array(num_of_words_list)
        if feature == 'logmelspec':
            assert doc_segment_spec.shape == (len(self.image_key_list)*5, self.padding_len, self.logmelspec_dim)
        elif feature == 'hubert':
            assert doc_segment_spec.shape == (len(self.image_key_list)*5, self.padding_len, self.hubert_dim)

        np.save(true_len_pth, num_of_words_list)
        np.save(seg_embed_pth, doc_segment_spec)

def main(args):
    basename = '-'.join(args.data_summary_json.split('-')[1:]).split('.')[0]
    print('processing %s' % basename)

    with open(args.data_summary_json, 'r') as f:
        data_summary = json.load(f)

    if args.data_split == 'val':
        val_json = data_summary['val']
        print('loading val_loader')
        val_writer = SummaryJsonReader(val_json, args.image_hdf5)
        #val_writer.write_image_to_file(op.join(args.output_dir, 'val_ims-' + basename + '.npy'))
        #val_writer.write_text_or_tree_to_file( \
        #    op.join(args.output_dir, 'val_ground-truth-' + basename + '.txt'), op.join(args.output_dir, 'val_caps-' + basename + '.txt'))
        val_writer.write_utterance_to_file( \
            op.join(args.output_dir, f'val_segment-{args.feature}_embed-' + basename + '.npy'), \
            op.join(args.output_dir, f'val_segment-{args.feature}_len-' + basename + '.npy'), feature=args.feature)

    if args.data_split == 'test':
        test_json = data_summary['test'] 
        print('loading test_loader')
        test_writer = SummaryJsonReader(test_json, args.image_hdf5)
        #test_writer.write_image_to_file(op.join(args.output_dir, 'test_ims-' + basename + '.npy'))
        #test_writer.write_text_or_tree_to_file( \
        #    op.join(args.output_dir, 'test_ground-truth-' + basename + '.txt'), op.join(args.output_dir, 'test_caps-' + basename + '.txt'))
        test_writer.write_utterance_to_file( \
            op.join(args.output_dir, f'test_segment-{args.feature}_embed-' + basename + '.npy'), \
            op.join(args.output_dir, f'test_segment-{args.feature}_len-' + basename + '.npy'), feature=args.feature)

    if args.data_split == 'train':
        train_json = data_summary['train']
        print('loading train_loader')
        train_writer = SummaryJsonReader(train_json, args.image_hdf5, parallelize=args.parallelize, num_labs=args.num_labs, lab_id=args.lab_id)
        #train_writer.write_image_to_file(op.join(args.output_dir, 'train_ims-' + basename + '.npy'))
        #train_writer.write_text_or_tree_to_file( \
        #    op.join(args.output_dir, 'train_ground-truth-' + basename + '.txt'), op.join(args.output_dir, 'train_caps-' + basename + '.txt'))
        
        if args.parallelize: 
            # first check if parallize lab files exist
            all_lab_embed_file, all_lab_len_file = [], []
            for tmp_lab_id in range(args.num_labs): 
                lab_embed_file = op.join(args.output_dir, f'.train_segment-{args.feature}_embed-' + basename + '-' + str(tmp_lab_id) + '.npy')
                lab_len_file = op.join(args.output_dir, f'.train_segment-{args.feature}_len-' + basename + '-' + str(tmp_lab_id) + '.npy')
                if op.exists(lab_embed_file) and op.exists(lab_len_file): 
                    parallelize_done = True 
                    all_lab_embed_file.append(lab_embed_file)
                    all_lab_len_file.append(lab_len_file)
                else: parallelize_done = False
            if parallelize_done: 
                print('Skip feature extraction. Combine all pre-extracted lab_embed_files.')
                train_writer.combine_lab_files(all_lab_embed_file, all_lab_len_file, \
                    op.join(args.output_dir, f'train_segment-{args.feature}_embed-' + basename + '.npy'), \
                    op.join(args.output_dir, f'train_segment-{args.feature}_len-' + basename + '.npy'), feature=args.feature)
            else: 
                print('write partial numpy arrays to lab file')
                lab_embed_file = op.join(args.output_dir, f'.train_segment-{args.feature}_embed-' + basename + '-' + str(args.lab_id) + '.npy')
                lab_len_file = op.join(args.output_dir, f'.train_segment-{args.feature}_len-' + basename + '-' + str(args.lab_id) + '.npy')
                if op.exists(lab_embed_file) and op.exists(lab_len_file):
                    print(f'{lab_embed_file} and {lab_len_file} already exist. Skip feature extraction')
                else: 
                    train_writer.write_utterance_to_file(lab_embed_file, lab_len_file, feature=args.feature)
        else: 
            train_writer.write_utterance_to_file( \
                op.join(args.output_dir, f'train_segment-{args.feature}_embed-' + basename + '.npy'), \
                op.join(args.output_dir, f'train_segment-{args.feature}_len-' + basename + '.npy'), feature=args.feature)

if __name__ == '__main__': 
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-summary-json', '-j', type=str)
    parser.add_argument('--image-hdf5', '-i', type=str)
    parser.add_argument('--output-dir', '-o', type=str)
    parser.add_argument('--parallelize', '-p', action="store_true")
    parser.add_argument('--num_labs', '-n', type=int)
    parser.add_argument('--lab_id', '-l', type=int)
    parser.add_argument('--data-split', '-s', type=str, choices = ['train', 'val', 'test'])
    parser.add_argument('--feature', '-f', type=str, default='logmelspec', 
                        choices = ['logmelspec', 'hubert'])
    args = parser.parse_args()

    main(args)
