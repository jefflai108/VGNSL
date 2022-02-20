import h5py

filename  = 'data/SpokenCOCO/Freda-formatting/val_segment-hubert8_embed-km2048-83k-5k-5k.hdf5'
tree_file = 'data/SpokenCOCO/Freda-formatting/val_ground-truth-83k-5k-5k.txt'
cap_file  = 'data/SpokenCOCO/Freda-formatting/val_caps-83k-5k-5k.txt'

def read_text(file_path):
    with open(file_path, 'r') as f: 
        content = f.readlines()
    content = [x.strip('\n') for x in content]
    return content 

def lossy_run_length_encoding(id_seq_list): 
    return [id for i, id in enumerate(id_seq_list) if i == 0 or id != id_seq_list[i-1]]

trees    = read_text(tree_file)
captions = read_text(cap_file)
features = h5py.File(filename, 'r')
for index in features.keys(): 
    print(index) 
    # extract discrete (speech) IDs from hdf5 file 
    discrete_feature_seq = features[index][:]
    # remove consequtive duplicate IDs
    deduplicated_discrete_feature_seq = lossy_run_length_encoding(discrete_feature_seq)
    # extract corresponding tree and text caption
    tree = trees[int(index)]
    cap  = captions[int(index)]
