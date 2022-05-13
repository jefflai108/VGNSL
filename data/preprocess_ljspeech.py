import os 
import string 
import shutil
import tqdm 

import benepar
import spacy
from nltk import Tree

def replace_leaves(tree, leaves):
    if isinstance(tree, str):
        return leaves[0]
    left = 0
    new_children = list()
    for child in tree:
        n_leaves = 1 if isinstance(child, str) else len(child.leaves())
        new_child = replace_leaves(child, leaves[left:left+n_leaves])
        new_children.append(new_child)
        left += n_leaves
    return Tree(tree.label(), new_children)

def remove_label(tree):
    if len(tree) == 1:
        if len(tree.leaves()) == 1:
            return tree.leaves()[0]
        return remove_label(tree[0])
    new_children = list()
    for child in tree:
        new_child = remove_label(child)
        new_children.append(new_child)
    return Tree('', new_children)

def _read_file(scp): 
    with open(scp, 'r') as f: 
        content = f.readlines()
    content = [x.strip('\n') for x in content]

    return content

def _write_to_file(string, fpath): 
    f = open(fpath, 'w') 
    f.write(string)
    f.close()

def prep_for_mfa(target_data_dir, wav_scp, text, ljspeech_download_dir): 
    # create directory contains {utt_id}.wav/{utt_id}.txt/{utt_id}-tree.txt for MFA 
    
    utt2wavpath = _read_file(wav_scp)
    utt2wavpath = {x.split()[0]:x.split()[1] for x in utt2wavpath}
    utt2transcript = _read_file(text)
    utt2transcript = {x.split()[0]:' '.join(x.split()[1:]) for x in utt2transcript}
    exclude = set(string.punctuation)
    
    # setup pre-trained English parser 
    nlp = spacy.load('en_core_web_md')
    if spacy.__version__.startswith('2'):
        nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
    else:
        nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    for utt in tqdm.tqdm(utt2wavpath.keys()): 
        full_wavpath = os.path.join(ljspeech_download_dir, utt2wavpath[utt])
        assert os.path.exists(full_wavpath)

        transcript = utt2transcript[utt].lower() # lower case
        transcript = ''.join(ch for ch in transcript if ch not in exclude) # remove puncs
        transcript = ' '.join(transcript.split()) # remove extra spaces 
    
        doc = nlp(transcript)
        sent = list(doc.sents)[0]
        tree = Tree.fromstring(sent._.parse_string)
        tree = remove_label(tree)
        tree = ' '.join(str(tree).replace('(', ' ( ').replace(')', ' ) ').split())

        shutil.copyfile(full_wavpath, os.path.join(target_data_dir, utt + '.wav'))
        _write_to_file(transcript, os.path.join(target_data_dir, utt + '.txt'))
        _write_to_file(tree, os.path.join(target_data_dir, utt + '-tree.txt'))

if __name__ == '__main__': 
    split = 'eval1'
    #split = 'dev'

    orig_data_dir = '/data/sls/scratch/clai24/syntax/VGNSL-feature/data/LJspeech/' + split 
    target_data_dir = '/data/sls/scratch/clai24/syntax/VGNSL-feature/data/LJspeech/' + split + '-speaker'
    ljspeech_download_dir = '/data/sls/temp/clai24/lottery-ticket/espnet/egs2/ljspeech/tts1'

    prep_for_mfa(target_data_dir, 
                 os.path.join(orig_data_dir, 'wav.scp'), 
                 os.path.join(orig_data_dir, 'text'), 
                 ljspeech_download_dir)
