import h5py
import os.path as osp
import random
import numpy as np 

def read_text(file_path):
    with open(file_path, 'r') as f: 
        content = f.readlines()
    content = [x.strip('\n') for x in content]
    return content 

def lossy_run_length_encoding(id_seq_list): 
    return [id for i, id in enumerate(id_seq_list) if i == 0 or id != id_seq_list[i-1]]

def slice_discrete_feature_seq(feat, word_list, pad_id=0, max_segment_len=50):
    """Given a sequence of feature IDs and word-level alignments, return a matrix of word-aligned sequence IDs. 
    the 1st dim of the matrix is padded to longest word segment for an given utterance
    
    input: (sequence_len,)
    output: (# of words, max_segment_len)
    """
    assert len(feat) >= round(word_list[-1][-1]), print(word_list, len(feat))
    word2len = [round(z)-round(y) for (_,y,z) in word_list]
    sliced_matrix = pad_id * np.ones((len(word_list), max_segment_len)) 

    for i, (word, start_frame, end_frame) in enumerate(word_list):
        start_frame, end_frame = round(start_frame), round(end_frame)
        if end_frame - start_frame > max_segment_len: 
            segment_len = max_segment_len
            start_frame = random.randint(start_frame, end_frame - max_segment_len)
            end_frame   = start_frame + max_segment_len  
        word_segment = feat[start_frame:end_frame]

        # remove consequtive duplicate IDs
        deduplicate_word_segment = lossy_run_length_encoding(word_segment)
        sliced_matrix[i, :len(deduplicate_word_segment)] = deduplicate_word_segment
    
    return sliced_matrix

def fetch_sample(features, alignments, captions, trees, index):
    # extract discrete (speech) IDs from hdf5 file 
    original_discrete_feature_seq = features[index][:]
    # extract corresponding alignment 
    alignment = alignments[int(index)]
    # apply word-level force alignment to the discrete ID sequence 
    aligned_discrete_feature_matrix = slice_discrete_feature_seq(original_discrete_feature_seq, 
                                                                 alignment)

    # extract corresponding tree and text caption
    tree = trees[int(index)]
    cap  = captions[int(index)]

    return (aligned_discrete_feature_matrix, tree, cap)

def iterate_samples(features, alignments, captions, trees): 
    for index in features.keys(): 
        # index for accessing feature/text cap/tree 
        print(index) 

        # fetch the samples correspond to the index 
        index_sample = fetch_sample(features, alignments, captions, trees, index)

if __name__ == '__main__':
    # specify filenames 
    data_directory = 'data/SpokenCOCO/Freda-formatting'
    feature_file   = osp.join(data_directory, 'val_segment-hubert8_embed-km2048-83k-5k-5k.hdf5')
    alignment_file = osp.join(data_directory, 'val_segment-hubert8_word_list-83k-5k-5k.npy')
    tree_file = osp.join(data_directory, 'val_ground-truth-83k-5k-5k.txt')
    cap_file  = osp.join(data_directory, 'val_caps-83k-5k-5k.txt')

    # prepare data 
    trees    = read_text(tree_file)
    captions = read_text(cap_file)
    features = h5py.File(feature_file, 'r')
    alignments = np.load(alignment_file, allow_pickle=True)[0]

    # iterate through sample 
    iterate_samples(features, alignments, captions, trees)
