import librosa
import numpy as np
import scipy.signal
import torch
import textgrid
import pickle
import os

import torch 
import torch.nn.functional as F
import s3prl.hub as hub

import w2v2_model_jeff

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

def read_textgrid(textgrid_pth, text_pth, nframes, frame_stride=0.01):
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
    
def slice_feature(logspec, word_list, 
                  target_sent_padded_length=None, sent_level_padding=False, 
                  target_segment_padded_length=None, return_whole=False):
    # for each word, compute the segment-level logmelspec by 
    #       default: averaging feature across (force-aligned) word segment --> sent_level_padding 
    #       alternative: return whole word segment --> sent_level_padding + segment_level_padding 
    if return_whole: 
        sentence_segment_spec = np.zeros((target_sent_padded_length, target_segment_padded_length, logspec.shape[0]))
    else: 
        if sent_level_padding:
            sentence_segment_spec = np.zeros((target_sent_padded_length, logspec.shape[0]))
        else: 
            sentence_segment_spec = np.zeros((len(word_list), logspec.shape[0]))

    segment_len_list = []
    for i, (word, start_frame, end_frame) in enumerate(word_list):
        if return_whole: 
            segment_spec = logspec[:, round(start_frame):round(end_frame)].T
            segment_len = round(end_frame)-round(start_frame)
            sentence_segment_spec[i, :segment_len] = segment_spec # store the whole feature segment 
            segment_len_list.append(segment_len)
        else: 
            segment_spec = np.mean(logspec[:, round(start_frame):round(end_frame)], axis=1) # averaged across feature segment
            sentence_segment_spec[i] = segment_spec
    
    return sentence_segment_spec, len(word_list), segment_len_list

def hubert_feature_extraction(input_utterance_pth, upstream_model, 
                              layer=12, hubert_dim=768, device='cpu'):
    # extract hubert feature on GPU with torch. 
    # return specified layer's representation: 0, 1, 2, ...., 12 (start with CNN encoder outputs)

    # load wavefile 
    y, native_sample_rate = librosa.load(input_utterance_pth)
    
    # resample to the target sample rate -- important for pre-trained 16k models
    y = librosa.resample(y, native_sample_rate, 16000)
    y = torch.from_numpy(y).to(device)

    with torch.no_grad():
        if hubert_dim == 1024: 
            # hubert_large; apply norm at input tensor 
            y = F.layer_norm(y, y.shape)
        reps = upstream_model([y])["hidden_states"]
        #print(f'reps has {len(reps)} layers')
        reps = reps[layer].squeeze().reshape(hubert_dim, -1)
        numpy_reps = reps.detach().cpu().numpy()
    n_frames = numpy_reps.shape[1]

    return numpy_reps, n_frames

def vghubert_feature_extraction(input_utterance_pth, model,
                                layer=11, device='cuda'):
    # extract vg-hubert feature on GPU with torch. 
    # return specified layer's representation: 0, 1, 2, ...., 11 (start with transformer_block1)

    # load wavefile 
    y, native_sample_rate = librosa.load(input_utterance_pth)
    
    # resample to the target sample rate -- important for pre-trained 16k models
    y = librosa.resample(y, native_sample_rate, 16000)
    y = torch.from_numpy(y).to(device)

    with torch.no_grad():
        # model forward pass
        w2v2_out = model(y.unsqueeze(0), 
                         padding_mask=None, 
                         mask=False, 
                         need_attention_weights=True, 
                         segment_layer=-1, 
                         feature_layer=[layer])

        # extract the representations
        reps = [item.squeeze(0)[1:] for item in w2v2_out['layer_features']] # [1, T+1, D] -> [T, D]
        reps = reps[0]
        numpy_reps = np.transpose(reps.detach().cpu().numpy())
    n_frames = numpy_reps.shape[1]

    return numpy_reps, n_frames

def setup_vg_hubert(model_type, snapshot='best', device='cuda'):
    # setup VG-Hubert based on Jason's code
    
    if model_type == 'disc-81':
        exp_dir = '/data/sls/scratch/clai24/data/SpokenCOCO/vghubert_model_weights/disc-81'
    elif model_type == 'disc-82':
        expdir = '/data/sls/scratch/clai24/data/SpokenCOCO/vghubert_model_weights/disc-82'
    else:
        print('%s not supported' % model_type)
        exit()

    with open(os.path.join(exp_dir, "args.pkl"), "rb") as f:
        model_args = pickle.load(f)
    model = w2v2_model_jeff.Wav2Vec2Model_cls(model_args)
    if "best" in snapshot:
        bundle = torch.load(os.path.join(exp_dir, "best_bundle.pth"))
    else:
        snapshot = int(snapshot)
        bundle = torch.load(os.path.join(exp_dir, f"snapshot_{snapshot}.pth"))

    if "dual_encoder" in bundle:
        model.carefully_load_state_dict(bundle['dual_encoder'], load_all=True)
    elif "audio_encoder" in bundle:
        model.carefully_load_state_dict(bundle['audio_encoder'], load_all=True)
    else:
        model.carefully_load_state_dict(bundle['model'], load_all=True)

    model.eval()
    model = model.to(device)

    return model

if __name__ == '__main__':
    wav_file = 'data/SpokenCOCO/wavs-speaker/m1vjq8cayvs6c9/m1vjq8cayvs6c9-32KTQ2V7RDFP25PU1AP8KXEZU8I9MO_92648_385961.wav'
    grid_file = 'data/SpokenCOCO/wavs-speaker-aligned/m1vjq8cayvs6c9/m1vjq8cayvs6c9-32KTQ2V7RDFP25PU1AP8KXEZU8I9MO_92648_385961.TextGrid'
    text_file = 'data/SpokenCOCO/wavs-speaker/m1vjq8cayvs6c9/m1vjq8cayvs6c9-32KTQ2V7RDFP25PU1AP8KXEZU8I9MO_92648_385961.txt'

    # example extracting segment-averaged logmelspec 
    frame_stride=0.01
    logspec, nframes = compute_spectrogram(wav_file) # (40, 530)
    word_list, word_string = read_textgrid(grid_file, text_file, nframes, frame_stride=frame_stride)
    sentence_segment_spec, num_of_words, _ = slice_feature(logspec, word_list, \
                                                           target_sent_padded_length=50, sent_level_padding=True, \
                                                           target_segment_padded_length=None, return_whole=False) # (50, 40)
    print(logspec.shape, nframes, sentence_segment_spec.shape, num_of_words) # (40, 530) 530 (50, 40) 12
    print('************************************')

    # example extracting whole-segmnet logmelspec 
    frame_stride=0.01
    logspec, nframes = compute_spectrogram(wav_file) # (40, 530)
    word_list, word_string = read_textgrid(grid_file, text_file, nframes, frame_stride=frame_stride)
    sentence_segment_spec, num_of_words, segment_len_list = slice_feature(logspec, word_list, \
                                                                          target_sent_padded_length=50, sent_level_padding=True, \
                                                                          target_segment_padded_length=int(7.90/frame_stride), return_whole=True) # (50, 790, 40)
    print(logspec.shape, nframes, sentence_segment_spec.shape, num_of_words) # (40, 530) 530 (50, 790, 40) 12
    print(word_list, word_string)
    print('************************************')

    # example extracting whole-segment hubert 
    # setup upstream model first 
    frame_stride=0.02
    HUBERT = getattr(hub, 'hubert_base')()
    HUBERT = getattr(hub, 'content_vec_v07_11')()
    HUBERT = getattr(hub, 'content_vec_v12_05')()
    # load pre-trained model 
    if torch.cuda.is_available(): 
        device = 'cuda'
    else: device = 'cpu'
    upstream_model = HUBERT.to(device)
    upstream_model = upstream_model.eval() # important -- this disables layerdrop of w2v2/hubert
    # upstream model feature extraction 
    hubert_repre, nframes = hubert_feature_extraction(wav_file, upstream_model, layer=12, device=device) # (768, 264)
    word_list, word_string = read_textgrid(grid_file, text_file, nframes, frame_stride=frame_stride)
    sentence_segment_spec, num_of_words, segment_len_list = slice_feature(hubert_repre, word_list, \
                                                                          target_sent_padded_length=50, sent_level_padding=True, \
                                                                          target_segment_padded_length=int(7.90/frame_stride), return_whole=True) # (50, 395, 768)
    print(hubert_repre.shape, nframes, sentence_segment_spec.shape, num_of_words, segment_len_list) # (768, 264) 264 (50, 395, 768) 12 [8, 19, 22, 8, 12, 8, 30, 19, 10, 24, 11, 46]
    print(word_list, word_string) # [('a', 35.0, 43.0), ('town', 43.0, 61.5), ('square', 61.5, 84.5), ('is', 84.5, 92.0), ('full', 92.0, 104.49999999999999), ('of', 104.49999999999999, 112.00000000000001), ('people', 112.00000000000001, 142.5), ('riding', 149.0, 167.5), ('their', 167.5, 178.0), ('bikes', 178.0, 201.5), ('and', 201.5, 213.49999999999997), ('skateboarding', 213.49999999999997, 259.0)] a town square is full of people riding their bikes and skateboarding
    print('************************************')

    # example extracting whole-segment VG-hubert 
    upstream_model = setup_vg_hubert(model_type='disc-81', snapshot='best', device=device)
    vghubert_repre, vghubert_nframes = vghubert_feature_extraction(wav_file, upstream_model,
                                                          layer=11, device=device)
    print(vghubert_repre.shape, nframes) # (768, 264) 264
    print('************************************')

    # example concatenating the extracted hubert and vghubert representations
    concat_hubert_and_vghubert_repre = np.concatenate((hubert_repre, vghubert_repre), axis=0)
    print(concat_hubert_and_vghubert_repre.shape) # (1536, 264)
