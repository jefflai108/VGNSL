import json 
import argparse 
from tqdm import tqdm
import os
from pathlib import Path
import subprocess
from collections import defaultdict 
import pickle

from create_vocab import build_vocab

# {"image": "val2014/COCO_val2014_000000325114.jpg", "captions": [{"text": "A URINAL IN A PUBLIC RESTROOM NEAR A WOODEN TABLE", "speaker": "m071506418gb9vo0w5xq3", "uttid": "m071506418gb9vo0w5xq3-3LUY3GC63Z0R9PYEETJGN5HO4UEP7B    _325114_629297", "wav": "wavs/val/0/m071506418gb9vo0w5xq3-3LUY3GC63Z0R9PYEETJGN5HO4UEP7B_325114_629297.wav", "parse_tree": "( ( a urinal ) ( in ( ( a public restroom ) ( near ( a wooden table ) ) ) ) )"}, {"text": "A WHITE URINAL WHIT    E TILES A RED TABLE WITH A DISH", "speaker": "m221bzbcdh4ro4", "uttid": "m221bzbcdh4ro4-3ZQIG0FLQEGJ4OWB8D0RLD5NKVLWVB_325114_628718", "wav": "wavs/val/0/m221bzbcdh4ro4-3ZQIG0FLQEGJ4OWB8D0RLD5NKVLWVB_325114_628718.wav", "parse_tree":     "( ( a white urinal white ) ( tiles ( a red table ) ( with ( a dish ) ) ) )"}, {"text": "A TABLE WITH A PLATE OF FOOD AND A URINAL BASIN", "speaker": "m2zco7272xa5e0", "uttid": "m2zco7272xa5e0-3RANCT1ZVFHR36908WUQ2DQJVTZBU1_325114_615    377", "wav": "wavs/val/0/m2zco7272xa5e0-3RANCT1ZVFHR36908WUQ2DQJVTZBU1_325114_615377.wav", "parse_tree": "( ( a table ) ( with ( ( ( a plate ) ( of food ) ) and ( a urinal basin ) ) ) )"}, {"text": "A LITTLE RED TABKE WITH A PLATE ON     IT IN THE BATHROOM", "speaker": "m2saqi3zjrms1y", "uttid": "m2saqi3zjrms1y-3F0BG9B9MPNLI3QF5GFZ0WA089EY7V_325114_630704", "wav": "wavs/val/0/m2saqi3zjrms1y-3F0BG9B9MPNLI3QF5GFZ0WA089EY7V_325114_630704.wav", "parse_tree": "( ( a little     red tabke ) ( with ( ( a plate ) ( on it ) ( in ( the bathroom ) ) ) ) )"}, {"text": "A URINAL NEXT TO A RED TABLE WITH A PLATE SITTING ON THE TABLE", "speaker": "ma7soiuo8jbku", "uttid": "ma7soiuo8jbku-3LOTDFNYA7ZU8RAL8YVN3R21Z9QFWI    _325114_622604", "wav": "wavs/val/0/ma7soiuo8jbku-3LOTDFNYA7ZU8RAL8YVN3R21Z9QFWI_325114_622604.wav", "parse_tree": "( ( a urinal ) ( next ( to ( ( a red table ) ( with ( ( a plate ) ( sitting ( on ( the table ) ) ) ) ) ) ) ) )"}]}
 
def create_speaker_subdirectory(args): 
    """ create a new wavs/ based on speaker split 
    """
    # collect spk2wav
    spk2wav = defaultdict(list)
    spk2wav = collect_spaker2wav(args.input_val_file, spk2wav)
    spk2wav = collect_spaker2wav(args.input_train_file, spk2wav)
    print('There are', len(spk2wav.keys()), ' speakers') # 2352

    # symbolic link to the new directory 
    for spk, wav_paths in tqdm(spk2wav.items()): 
        new_spk_dir = os.path.join(args.data_directory, args.new_data_directory_name, spk)
        Path(new_spk_dir).mkdir(parents=True, exist_ok=True)
        for orig_wav_path in wav_paths: 
            orig_full_path = os.path.join(os.getcwd(), orig_wav_path) # use full path
            assert os.path.exists(orig_full_path), print(orig_full_path)
            bashCommand = "ln -s " + orig_full_path + " " + new_spk_dir
            print(bashCommand)
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()

def collect_spaker2wav(input_file, spk2wav = {}): 
    data = json.load(open(input_file))
    for pair in tqdm(data['data']):
        img = pair['image']
        captions = pair['captions']
        for caption in captions: # each image pairs with 5 captions
            speaker = caption['speaker']
            wav_path = os.path.join(args.data_directory, caption['wav'])
            spk2wav[speaker].append(wav_path)

    return spk2wav

def export_transcript_and_tree(args):
    cnt_val = _export_transcript_and_tree(args.input_val_file)
    cnt_train = _export_transcript_and_tree(args.input_train_file)

    print('There are %d utterances in total' % (cnt_val + cnt_train)) # There are 617222 utterances in total

def _export_transcript_and_tree(input_file): 
    data = json.load(open(input_file))
  
    cnt = 0
    for pair in tqdm(data['data']):
        img = pair['image']
        captions = pair['captions']
        for caption in captions: # each image pairs with 5 captions
            speaker = caption['speaker']
            wav_id = os.path.join(caption['wav']).split('/')[-1]
            wav_path = os.path.join(args.data_directory, args.new_data_directory_name, speaker, wav_id)
            transcript = caption['text'].lower()
            parse_tree = caption['parse_tree']
            _write_to_file(transcript, wav_path.replace('.wav', '.txt'))
            _write_to_file(parse_tree, wav_path.replace('.wav', '-tree.txt'))
            cnt += 1

    return cnt

def _write_to_file(string, fpath): 
    f = open(fpath, 'w') 
    f.write(string)
    f.close()

def _collect_transcript(input_file, transcripts): 
    data = json.load(open(input_file))

    for pair in tqdm(data['data']):
        img = pair['image']
        captions = pair['captions']
        for caption in captions: # each image pairs with 5 captions
            transcripts.append(caption['text'].lower())

    return transcripts

def create_and_store_vocab(args):
    transcripts = []
    transcripts = _collect_transcript(args.input_val_file, transcripts)
    transcripts = _collect_transcript(args.input_train_file, transcripts)

    vocab = build_vocab(transcripts, threshold=4)
    with open(args.vocab_output_pickle, 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
    print("Saved vocabulary file to ", args.vocab_output_pickle)

if __name__ == '__main__':
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-val-file', '-v', type=str)
    parser.add_argument('--input-train-file', '-t', type=str)
    parser.add_argument('--data-directory', '-d', type=str)
    parser.add_argument('--new-data-directory-name', '-n', type=str)
    parser.add_argument('--vocab-output-pickle', '-p', type=str)
    args = parser.parse_args()

    # Step 1: create speaker-split wavs directory with softlink
    #create_speaker_subdirectory(args)

    # Step 2: export text transcript and parse tree to the new wavs directory
    #export_transcript_and_tree(args) 
    
    # Step 3: create vocab dictionary based on the transcripts
    #create_and_store_vocab(args)
