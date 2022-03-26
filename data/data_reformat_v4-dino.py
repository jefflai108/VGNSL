import nltk
import numpy as np
import os.path as op
import json 
import h5py
from tqdm import tqdm
import argparse
import torch 
import s3prl.hub as hub

from utils import compute_spectrogram, read_textgrid, slice_feature, hubert_feature_extraction

class SummaryJsonReader(object):
    def __init__(self, data_summary_json, image_hdf5, feature, img_embed_dim=2048,
                 parallelize=False, num_labs=None, lab_id=None):

        self.image_h5 = h5py.File(image_hdf5, 'r')
        self.data_summary_json = data_summary_json
        self.img_embed_dim = img_embed_dim

        # for parallel extraction 
        self.num_labs = num_labs

        # for speech features
        self.logmelspec_dim = 40
        if feature == 'hubert_large':
            self.hubert_dim = 1024
        elif feature == 'hubert' or feature == 'content_vec_v07_11' or feature == 'content_vec_v12_05':
            self.hubert_dim = 768
        self.logmelspec_frame_stride = 0.01
        self.hubert_frame_stride = 0.02
        self.sent_level_padding_len = 50
        self.logmelspec_seg_level_padding_len = int(7.90/self.logmelspec_frame_stride)
        self.hubert_seg_level_padding_len = int(7.90/self.hubert_frame_stride)

        # set device 
        if torch.cuda.is_available(): 
            self.device = 'cuda'
        else: self.device = 'cpu'

        # init upstream model for feat extraction 
        if feature == 'hubert_large':
            self.init_hubert_large()
        elif feature == 'hubert':
            self.init_hubert()
        elif feature == 'content_vec_v07_11' or feature == 'content_vec_v12_05': 
            self.init_content_vec(feature)
        
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

    def init_hubert_large(self): 
        HUBERT = getattr(hub, 'hubert_large_ll60k')()
        # load pre-trained model 
        upstream_model = HUBERT.to(self.device)
        self.upstream_model = upstream_model.eval() # important -- this disables layerdrop of w2v2/hubert

    def init_hubert(self): 
        HUBERT = getattr(hub, 'hubert_base')()
        # load pre-trained model 
        upstream_model = HUBERT.to(self.device)
        self.upstream_model = upstream_model.eval() # important -- this disables layerdrop of w2v2/hubert

    def init_content_vec(self, feature): 
        HUBERT = getattr(hub, feature)()
        # load pre-trained model 
        upstream_model = HUBERT.to(self.device)
        self.upstream_model = upstream_model.eval() # important -- this disables layerdrop of w2v2/hubert

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
            assert doc_segment_spec.shape == (len(self.image_key_list)*5, self.sent_level_padding_len, self.logmelspec_dim)
        elif feature == 'hubert' or feature == 'content_vec_v07_11' or feature == 'content_vec_v12_05':
            assert doc_segment_spec.shape == (len(self.image_key_list)*5, self.sent_level_padding_len, self.hubert_dim)

        np.save(true_len_pth, num_of_words_list)
        np.save(seg_embed_pth, doc_segment_spec)

def main(args):
    basename = '-'.join(args.data_summary_json.split('-')[1:]).split('.')[0]
    print('processing %s' % basename)

    with open(args.data_summary_json, 'r') as f:
        data_summary = json.load(f)
   
    if args.dino_type in ['vits8', 'vits16']: 
        dino_dim = 384
    elif args.dino_type in ['vitb8', 'vitb16']: 
        dino_dim = 768

    if args.data_split == 'val':
        val_json = data_summary['val'] 
        print('loading val_loader')
        val_writer = SummaryJsonReader(val_json, args.image_hdf5, args.feature, img_embed_dim=dino_dim)
        val_writer.write_image_to_file(op.join(args.output_dir, 'val_' + args.dino_type + '_ims-' + basename + '.npy'))
  
    if args.data_split == 'test':
        test_json = data_summary['test'] 
        print('loading test_loader')
        test_writer = SummaryJsonReader(test_json, args.image_hdf5, args.feature, img_embed_dim=dino_dim)
        test_writer.write_image_to_file(op.join(args.output_dir, 'test_' + args.dino_type + '_ims-' + basename + '.npy'))
   
    if args.data_split == 'train':
        train_json = data_summary['train']
        print('loading train_loader')
        train_writer = SummaryJsonReader(train_json, args.image_hdf5, args.feature, parallelize=args.parallelize, num_labs=args.num_labs, lab_id=args.lab_id, img_embed_dim=dino_dim)
        train_writer.write_image_to_file(op.join(args.output_dir, 'train_' + args.dino_type + '_ims-' + basename + '.npy'))
           
    
if __name__ == '__main__': 
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-summary-json', '-j', type=str)
    parser.add_argument('--image-hdf5', '-i', type=str)
    parser.add_argument('--output-dir', '-o', type=str)
    parser.add_argument('--parallelize', '-p', action="store_true")
    parser.add_argument('--return_whole', '-r', action="store_true")
    parser.add_argument('--h5_format', action="store_true")
    parser.add_argument('--num_labs', '-n', type=int)
    parser.add_argument('--lab_id', '-l', type=int)
    parser.add_argument('--data-split', '-s', type=str, choices = ['train', 'val', 'test'])
    parser.add_argument('--feature', '-f', type=str, default='logmelspec', 
                        choices = ['logmelspec', 'hubert', 'hubert_large', 'content_vec_v07_11', 'content_vec_v12_05'])
    parser.add_argument('--layer_num', type=int, default=12)
    parser.add_argument('--dino_type', type=str, choices=['vits8', 'vits16', 'vitb8', 'vitb16'])
    args = parser.parse_args()

    main(args)
