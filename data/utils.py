# Author: Wei-Ning Hsu
import librosa
import numpy as np
import scipy.signal
import torch
import textgrid

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
    '''
    if use_raw_length:
        target_length = n_frames
    p = target_length - n_frames
    if p > 0:
        logspec = np.pad(logspec, ((0,0),(0,p)), 'constant',
                         constant_values=(padval,padval))
    elif p < 0:
        print('WARNING: truncate %d/%d frames' % (-p, n_frames))
        logspec = logspec[:,0:p]
        n_frames = target_length
    '''

    #logspec = torch.FloatTensor(logspec)
    return logspec, n_frames

def read_textgrid(textgrid_pth, text_pth, nframes, frame_stride=0.01):
    # return [(0-th word, start frame, end frame), ..., (n-th word, start frame, end frame)]
    word_tgs = textgrid.TextGrid.fromFile(textgrid_pth)[0]

    word_list = []
    word_string = []
    for word_tg in word_tgs: 
        word = word_tg.mark
        if word == '': # probably silence
            continue 
        # convert to frame-based (0.01s stride)
        word_obj = (word, word_tg.minTime/frame_stride, word_tg.maxTime/frame_stride)
        word_list.append(word_obj)
        word_string.append(word)
    assert word_tg.maxTime <= nframes

    word_string = ' '.join(word_string)
    with open(text_pth, 'r') as f: 
        gt_text = f.readline()
    assert gt_text == word_string, print(f'{word_string}\n{gt_text}')

    return word_list, word_string
    
def slice_spectrogram(logspec, word_list, target_padded_length=None, padding=False):
    # for each word, compute the segment-level logmelspec by averaging feature across (force-aligned) word segment
    # (optional) pad 0s in the *word* dimension 
    if not padding:
        sentence_segment_spec = np.zeros((len(word_list), logspec.shape[0]))
    else: sentence_segment_spec = np.zeros((target_padded_length, logspec.shape[0]))

    for i, (word, start_frame, end_frame) in enumerate(word_list):
        semgnet_spec = np.mean(logspec[:, round(start_frame):round(end_frame)], axis=1) # averaged across frame dimension 
        sentence_segment_spec[i] = semgnet_spec

    return sentence_segment_spec, len(word_list)
        

if __name__ == '__main__':
    # example usage
    wav_file = 'data/SpokenCOCO/wavs-speaker/m1vjq8cayvs6c9/m1vjq8cayvs6c9-32KTQ2V7RDFP25PU1AP8KXEZU8I9MO_92648_385961.wav'
    grid_file = 'data/SpokenCOCO/wavs-speaker-aligned/m1vjq8cayvs6c9/m1vjq8cayvs6c9-32KTQ2V7RDFP25PU1AP8KXEZU8I9MO_92648_385961.TextGrid'
    text_file = 'data/SpokenCOCO/wavs-speaker/m1vjq8cayvs6c9/m1vjq8cayvs6c9-32KTQ2V7RDFP25PU1AP8KXEZU8I9MO_92648_385961.txt'
    logspec, nframes = compute_spectrogram(wav_file) # torch.Size([40, 530])
    word_list, word_string = read_textgrid(grid_file, text_file, nframes)
    sentence_segment_spec, num_of_words = slice_spectrogram(logspec, word_list, 20, True)
