import nltk
import numpy as np
import os.path as op
import json 
import h5py
from tqdm import tqdm
import argparse
import torch 
import s3prl.hub as hub

from utils import slice_feature, hubert_feature_extraction, 
                  setup_vg_hubert, vghubert_feature_extraction

class SummaryJsonReader(object):
    def __init__(self, data_summary_json, feature,
                 parallelize=False, num_labs=None, lab_id=None):

        self.data_summary_json = data_summary_json

        # for parallel extraction 
        self.num_labs = num_labs

        # for speech features
        self.hubert_dim = 768
        self.vghubert_dim = 768

        # set device 
        if torch.cuda.is_available(): 
            self.device = 'cuda'
        else: self.device = 'cpu'

        # init upstream model for feat extraction 
        if 'hubert2' in feature:
            self.init_hubert()
        self.upstream_vghubert = setup_vg_hubert(model_type=feature, snapshot='best', device=self.device)

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

    def write_utterance_to_h5(self, seg_embed_h5_obj, word_list_pth, feature='disc-81', vghubert_layer_num=11):
        print('writing to h5 object')
        # data structure: h5py_obj[str(index)][:] = np.array(feature)
        # no splicing -- do it on-the-fly 

        word_list_dict = {}
        idx = 0
        for image_key in tqdm(self.utt_image_key_list):
            captions_list = self.data_summary_json[image_key]
            captions_list = self.__deduplicate__(captions_list, image_key)
            for captions in captions_list: 
                wav_file = captions[0]
           
                if 'hubert2' in feature:
                    hubert_repre, hubert_nframes = hubert_feature_extraction(wav_file, self.upstream_hubert, hubert_dim=self.hubert_dim, layer=2, device=self.device) # (768, 511)
                vghubert_repre, vghubert_nframes = vghubert_feature_extraction(wav_file, self.upstream_vghubert, layer=layer_num, device=self.device)
                print(hubret_repre.shape, hubert_nframes)
                print(vghubert_repre.shape, vghubert_nframes) # (768, 264) 264
                print(layer_num, type(vghubert_repre)) # <class 'numpy.ndarray'> (768, 511) 511

                assert hubert_nframes == vghubert_nframes

                if 'hubert2' in feature:
                    final_repre = np.concatenate((hubert_repre, vghubert_repre), dim=0)
                else: final_repre = vghubert_repre
                print(final_repre.shape)

                seg_embed_h5_obj.create_dataset(str(idx), data=final_repre.T)
                idx += 1

    def init_hubert(self): 
        HUBERT = getattr(hub, 'hubert_base')()
        # load pre-trained model 
        upstream_model = HUBERT.to(self.device)
        self.upstream_hubert = upstream_model.eval() # important -- this disables layerdrop of w2v2/hubert

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

def main(args):
    basename = '-'.join(args.data_summary_json.split('-')[1:]).split('.')[0]
    print('processing %s' % basename)

    with open(args.data_summary_json, 'r') as f:
        data_summary = json.load(f)
   
    if args.data_split == 'val':
        val_json = data_summary['val'] 
        print('loading val_loader')
        val_writer = SummaryJsonReader(val_json, args.feature)
        if args.h5_format:
            seg_embed_h5_obj = h5py.File(op.join(args.output_dir, f'val_segment-{args.feature}{args.layer_num}_embed-' + basename + '.hdf5'), "w")
            word_list_file = op.join(args.output_dir, f'val_segment-{args.feature}{args.layer_num}_word_list-' + basename + '.npy')
            val_writer.write_utterance_to_h5(seg_embed_h5_obj, word_list_file, feature=args.feature, layer_num=args.layer_num)
            seg_embed_h5_obj.close()
    
    if args.data_split == 'test':
        test_json = data_summary['test'] 
        print('loading test_loader')
        test_writer = SummaryJsonReader(test_json, args.feature)
        if args.h5_format:
            seg_embed_h5_obj = h5py.File(op.join(args.output_dir, f'test_segment-{args.feature}{args.layer_num}_embed-' + basename + '.hdf5'), "w")
            word_list_file = op.join(args.output_dir, f'test_segment-{args.feature}{args.layer_num}_word_list-' + basename + '.npy')
            test_writer.write_utterance_to_h5(seg_embed_h5_obj, word_list_file, feature=args.feature, layer_num=args.layer_num)
            seg_embed_h5_obj.close()

    if args.data_split == 'train':
        train_json = data_summary['train']
        print('loading train_loader')
        train_writer = SummaryJsonReader(train_json, args.feature, parallelize=args.parallelize, num_labs=args.num_labs, lab_id=args.lab_id)
        
        if args.parallelize: 
            if args.h5_format:
                # first check if parallize lab files exist
                all_lab_embed_file, all_lab_len_file = [], []
                for tmp_lab_id in range(args.num_labs): 
                    lab_seg_embed_h5_obj = op.join(args.output_dir, f'.train_segment-{args.feature}{args.layer_num}_embed-' + basename + '-' + str(tmp_lab_id) + '.hdf5')
                    lab_word_list_file = op.join(args.output_dir, f'.train_segment-{args.feature}{args.layer_num}_word_list-' + basename + '-' + str(tmp_lab_id) + '.npy')
                    if op.exists(lab_seg_embed_h5_obj) and op.exists(lab_word_list_file): 
                        parallelize_done = True 
                        all_lab_embed_file.append(h5py.File(lab_seg_embed_h5_obj, "r"))
                        all_lab_len_file.append(lab_word_list_file)
                    else: parallelize_done = False
                if parallelize_done: 
                    print('Skip feature extraction. Combine all pre-extracted lab_embed_files.')
                    train_writer.combine_h5_files(all_lab_embed_file, all_lab_len_file, \
                        h5py.File(op.join(args.output_dir, f'train_segment-{args.feature}{args.layer_num}_embed-' + basename + '.hdf5'), 'w'), \
                        op.join(args.output_dir, f'train_segment-{args.feature}{args.layer_num}_word_list-' + basename + '.npy'), \
                        feature=args.feature)
                    for lab_seg_embed_h5_obj in all_lab_embed_file: 
                        lab_seg_embed_h5_obj.close()
                else: 
                    print('write partial numpy arrays to lab file')
                    lab_seg_embed_h5_obj = op.join(args.output_dir, f'.train_segment-{args.feature}{args.layer_num}_embed-' + basename + '-' + str(args.lab_id) + '.hdf5')
                    lab_word_list_file = op.join(args.output_dir, f'.train_segment-{args.feature}{args.layer_num}_word_list-' + basename + '-' + str(args.lab_id) + '.npy')
                    if op.exists(lab_seg_embed_h5_obj) and op.exists(lab_word_list_file):
                        print(f'{lab_seg_embed_h5_obj} and {lab_word_list_file} already exist. Skip feature extraction')
                    else: 
                        lab_seg_embed_h5_obj = h5py.File(lab_seg_embed_h5_obj, "w")
                        train_writer.write_utterance_to_h5(lab_seg_embed_h5_obj, lab_word_list_file, feature=args.feature, layer_num=args.layer_num)
                        lab_seg_embed_h5_obj.close()
        else: 
            if args.h5_format:
                seg_embed_h5_obj = h5py.File(op.join(args.output_dir, f'train_segment-{args.feature}{args.layer_num}_embed-' + basename + '.hdf5'), "w")
                word_list_file = op.join(args.output_dir, f'train_segment-{args.feature}{args.layer_num}_word_list-' + basename + '.npy')
                train_writer.write_utterance_to_h5(seg_embed_h5_obj, word_list_file, feature=args.feature, layer_num=args.layer_num)
                seg_embed_h5_obj.close()

if __name__ == '__main__': 
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-summary-json', '-j', type=str)
    parser.add_argument('--output-dir', '-o', type=str)
    parser.add_argument('--parallelize', '-p', action="store_true")
    parser.add_argument('--h5_format', action="store_true")
    parser.add_argument('--num_labs', '-n', type=int)
    parser.add_argument('--lab_id', '-l', type=int)
    parser.add_argument('--data-split', '-s', type=str, choices = ['train', 'val', 'test'])
    parser.add_argument('--feature', '-f', type=str, default='disc-81', 
                        choices = ['disc-81_cat_hubert2', 'disc-82_cat_hubert2', 'disc-81', 'disc-82'])
    parser.add_argument('--layer_num', type=int, default=12)
    args = parser.parse_args()

    main(args)
