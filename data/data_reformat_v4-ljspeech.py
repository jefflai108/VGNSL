import nltk
import numpy as np
import os
import os.path as op
import json 
import h5py
from tqdm import tqdm
import argparse
import torch 
import s3prl.hub as hub

from utils import read_textgrid, hubert_feature_extraction

class SummaryJsonReader(object):
    def __init__(self, data_dir, data_split, feature):

        # set paths 
        wav_scp = os.path.join(data_dir, data_split, 'wav.scp')
        self.main_data_dir = os.path.join(data_dir, data_split + '-speaker')
        self.mfa_data_dir = os.path.join(data_dir, data_split + '-speaker-aligned')

        # for speech features
        if feature == 'hubert_large':
            self.hubert_dim = 1024
        elif feature == 'hubert':
            self.hubert_dim = 768
        self.hubert_frame_stride = 0.02

        # set device 
        if torch.cuda.is_available(): 
            self.device = 'cuda'
        else: self.device = 'cpu'

        # init upstream model for feat extraction 
        if feature == 'hubert_large':
            self.init_hubert_large()
        elif feature == 'hubert':
            self.init_hubert()
        
        print('pre-store ordered utt_id to ensure the storing order is consistent')
        with open(wav_scp, 'r') as f: 
            uttids = f.readlines()
        self.uttid_list = [x.strip('\n').split()[0] for x in uttids]

    def write_text_or_tree_to_file(self, tree_pth, transcript_pth): 
        print('writing tree to %s and transcript to %s' % (tree_pth, transcript_pth))
        tree_f = open(tree_pth, 'w')
        transcript_f = open(transcript_pth, 'w')
            
        for uttid in tqdm(self.uttid_list):
            transcript_file = os.path.join(self.main_data_dir, uttid + '.txt')
            tree_file = os.path.join(self.main_data_dir, uttid + '-tree.txt')

            transcript_f.write('%s\n' % self.__readfile__(transcript_file))
            tree_f.write('%s\n' % self.__readfile__(tree_file))
    
    @staticmethod
    def __readfile__(fpath): 
        with open(fpath, 'r') as f: 
            string = f.readline()
        return string 

    def write_utterance_to_h5(self, seg_embed_h5_obj, word_list_pth, feature='logmelspec', layer_num=12):
        print('writing to h5 object')
        # data structure: h5py_obj[str(index)][:] = np.array(feature)
        # no splicing -- do it on-the-fly 

        word_list_dict = {}
        idx = 0
        for uttid in tqdm(self.uttid_list):
            transcript_file = os.path.join(self.main_data_dir, uttid + '.txt')
            wav_file = os.path.join(self.main_data_dir, uttid + '.wav')
            alignment_file = os.path.join(self.mfa_data_dir, uttid + '.TextGrid')

            feat, nframes = hubert_feature_extraction(wav_file, self.upstream_model, hubert_dim=self.hubert_dim, layer=layer_num, device=self.device) # (768, 511)
            word_list, word_string = read_textgrid(alignment_file, transcript_file, nframes, frame_stride=self.hubert_frame_stride)

            #print(layer_num, type(feat), feat.shape, nframes) # <class 'numpy.ndarray'> (768, 511) 511
            word_list_dict[idx] = word_list
            seg_embed_h5_obj.create_dataset(str(idx), data=feat.T)
            idx += 1

        np.save(word_list_pth, [word_list_dict])

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

def main(args):
    basename = 'LJ' + args.data_split 
    print('processing %s' % basename)

    writer = SummaryJsonReader(args.data_dir, args.data_split, args.feature)
    writer.write_text_or_tree_to_file(
        op.join(args.output_dir, f'{args.data_split}_ground-truth-{basename}.txt'), 
        op.join(args.output_dir, f'{args.data_split}_caps-{basename}.txt'))
    
    if args.h5_format:
        seg_embed_h5_obj = h5py.File(op.join(args.output_dir, f'{args.data_split}_segment-{args.feature}{args.layer_num}_embed-{basename}.hdf5'), "w")
        word_list_file = op.join(args.output_dir, f'{args.data_split}_segment-{args.feature}{args.layer_num}_word_list-{basename}.npy')
        writer.write_utterance_to_h5(seg_embed_h5_obj, word_list_file, feature=args.feature, layer_num=args.layer_num)
        seg_embed_h5_obj.close()

if __name__ == '__main__': 
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/LJspeech', type=str)
    parser.add_argument('--output-dir', '-o', type=str)
    parser.add_argument('--h5_format', action="store_true")
    parser.add_argument('--data-split', '-s', type=str, choices = ['dev', 'eval1'])
    parser.add_argument('--feature', '-f', type=str, default='hubert', 
                        choices = ['hubert', 'hubert_large'])
    parser.add_argument('--layer_num', type=int, default=12)
    args = parser.parse_args()

    main(args)
