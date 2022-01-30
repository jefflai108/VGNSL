import nltk
import numpy as np
import os.path as op
import json 
import h5py
from tqdm import tqdm
import argparse
import textgrid
import math
import librosa
import scipy

def read_textgrid(textgrid_pth, text_pth, frame_stride=0.01):
    # return [(0-th word, start frame, end frame), ..., (n-th word, start frame, end frame)]
    # 
    # note: logmelspec has frame_stride 10ms, while SSL models like hubert has 20ms 
    word_tgs = textgrid.TextGrid.fromFile(textgrid_pth)[0]

    word_list = []
    word_string = []
    max_word_duration = 0
    for word_tg in word_tgs: 
        word = word_tg.mark
        if word == '': # probably silence
            continue 
        # convert to frame-based (0.01s/0.02s stride)
        word_duration = math.ceil(word_tg.maxTime/frame_stride - word_tg.minTime/frame_stride)
        max_word_duration = max(max_word_duration, word_duration)
   
    return max_word_duration # in frames 
 
WINDOWS = {'hamming': scipy.signal.hamming,
           'hann': scipy.signal.hann,
           'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}

def compute_spectrogram(input_utterance_pth, audio_conf={}):
    # load wavefile 
    y, native_sample_rate = librosa.load(input_utterance_pth)

    # Default audio configuration
    audio_type = audio_conf.get('audio_type', 'melspectrogram')
    if audio_type not in ['melspectrogram', 'spectrogram']:
        raise ValueError('Invalid audio_type specified in audio_conf.')

    preemph_coef = audio_conf.get('preemph_coef', 0.97)
    sample_rate = audio_conf.get('sample_rate', 16000)
    window_size = audio_conf.get('window_size', 0.025)
    window_stride = audio_conf.get('window_stride', 0.01)
    window_type = audio_conf.get('window_type', 'hamming')
    n_fft = audio_conf.get('n_fft', int(sample_rate * window_size))

    num_mel_bins = audio_conf.get('num_mel_bins', 40)
    fmin = audio_conf.get('fmin', 20)

    target_length = audio_conf.get('target_length', 2048)
    use_raw_length = audio_conf.get('use_raw_length', True)
    padval = audio_conf.get('padval', 0)
    
    win_length = int(sample_rate * window_size)
    hop_length = int(sample_rate * window_stride)

    # resample to the target sample rate
    y = librosa.resample(y, native_sample_rate, sample_rate)

    # subtract DC, preemphasis
    if y.size == 0:
        y = np.zeros(200)
    y = y - y.mean()
    y = np.append(y[0],y[1:]-preemph_coef*y[:-1])
    
    # compute spectrogram
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                        win_length=win_length,
                        window=WINDOWS[window_type])
    spec = np.abs(stft)**2
    if audio_type == 'melspectrogram':
        mel_basis = librosa.filters.mel(sample_rate, n_fft, n_mels=num_mel_bins,
                                        fmin=fmin)
        spec = np.dot(mel_basis, spec)
    logspec = librosa.power_to_db(spec, ref=np.max)

    # optional trimming/padding
    n_frames = logspec.shape[1]

    return logspec, n_frames


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

        max_sentence_word_duration = 0
        for image_key in tqdm(self.utt_image_key_list):
            captions_list = self.data_summary_json[image_key]
            captions_list = self.__deduplicate__(captions_list, image_key)
            for captions in captions_list: 
                transcript_file = captions[1]
                wav_file = captions[0]
                alignment_file = captions[3]
            
                if feature == 'logmelspec':
                    feat, nframes = compute_spectrogram(wav_file) # (40, 530)
                    #sentence_word_duration = read_textgrid(alignment_file, transcript_file, frame_stride=0.01)
                elif feature == 'hubert':
                    continue
                    #sentence_word_duration = read_textgrid(alignment_file, transcript_file, frame_stride=0.02)

                #max_sentence_word_duration = max(max_sentence_word_duration, sentence_word_duration)
                max_sentence_word_duration = max(max_sentence_word_duration, nframes)
                #print(f'max_sentence_word_duration is {max_sentence_word_duration}')
                print(f'max number of frames is {max_sentence_word_duration}')
                

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
        val_writer.write_utterance_to_file( \
            op.join(args.output_dir, f'val_segment-{args.feature}_embed-' + basename + '.npy'), \
            op.join(args.output_dir, f'val_segment-{args.feature}_len-' + basename + '.npy'), feature=args.feature)

    if args.data_split == 'test':
        test_json = data_summary['test'] 
        print('loading test_loader')
        test_writer = SummaryJsonReader(test_json, args.image_hdf5)
        test_writer.write_utterance_to_file( \
            op.join(args.output_dir, f'test_segment-{args.feature}_embed-' + basename + '.npy'), \
            op.join(args.output_dir, f'test_segment-{args.feature}_len-' + basename + '.npy'), feature=args.feature)

    if args.data_split == 'train':
        train_json = data_summary['train']
        print('loading train_loader')
        train_writer = SummaryJsonReader(train_json, args.image_hdf5, parallelize=args.parallelize, num_labs=args.num_labs, lab_id=args.lab_id)
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
