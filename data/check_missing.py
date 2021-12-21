import json 
import argparse 
from tqdm import tqdm
import os
from os.path import exists
from collections import defaultdict 
import wave
import contextlib

# {"image": "val2014/COCO_val2014_000000325114.jpg", "captions": [{"text": "A URINAL IN A PUBLIC RESTROOM NEAR A WOODEN TABLE", "speaker": "m071506418gb9vo0w5xq3", "uttid": "m071506418gb9vo0w5xq3-3LUY3GC63Z0R9PYEETJGN5HO4UEP7B    _325114_629297", "wav": "wavs/val/0/m071506418gb9vo0w5xq3-3LUY3GC63Z0R9PYEETJGN5HO4UEP7B_325114_629297.wav", "parse_tree": "( ( a urinal ) ( in ( ( a public restroom ) ( near ( a wooden table ) ) ) ) )"}, {"text": "A WHITE URINAL WHIT    E TILES A RED TABLE WITH A DISH", "speaker": "m221bzbcdh4ro4", "uttid": "m221bzbcdh4ro4-3ZQIG0FLQEGJ4OWB8D0RLD5NKVLWVB_325114_628718", "wav": "wavs/val/0/m221bzbcdh4ro4-3ZQIG0FLQEGJ4OWB8D0RLD5NKVLWVB_325114_628718.wav", "parse_tree":     "( ( a white urinal white ) ( tiles ( a red table ) ( with ( a dish ) ) ) )"}, {"text": "A TABLE WITH A PLATE OF FOOD AND A URINAL BASIN", "speaker": "m2zco7272xa5e0", "uttid": "m2zco7272xa5e0-3RANCT1ZVFHR36908WUQ2DQJVTZBU1_325114_615    377", "wav": "wavs/val/0/m2zco7272xa5e0-3RANCT1ZVFHR36908WUQ2DQJVTZBU1_325114_615377.wav", "parse_tree": "( ( a table ) ( with ( ( ( a plate ) ( of food ) ) and ( a urinal basin ) ) ) )"}, {"text": "A LITTLE RED TABKE WITH A PLATE ON     IT IN THE BATHROOM", "speaker": "m2saqi3zjrms1y", "uttid": "m2saqi3zjrms1y-3F0BG9B9MPNLI3QF5GFZ0WA089EY7V_325114_630704", "wav": "wavs/val/0/m2saqi3zjrms1y-3F0BG9B9MPNLI3QF5GFZ0WA089EY7V_325114_630704.wav", "parse_tree": "( ( a little     red tabke ) ( with ( ( a plate ) ( on it ) ( in ( the bathroom ) ) ) ) )"}, {"text": "A URINAL NEXT TO A RED TABLE WITH A PLATE SITTING ON THE TABLE", "speaker": "ma7soiuo8jbku", "uttid": "ma7soiuo8jbku-3LOTDFNYA7ZU8RAL8YVN3R21Z9QFWI    _325114_622604", "wav": "wavs/val/0/ma7soiuo8jbku-3LOTDFNYA7ZU8RAL8YVN3R21Z9QFWI_325114_622604.wav", "parse_tree": "( ( a urinal ) ( next ( to ( ( a red table ) ( with ( ( a plate ) ( sitting ( on ( the table ) ) ) ) ) ) ) ) )"}]}

   
# {"filepath": "val2014", "sentids": [770337, 771687, 772707, 776154, 781998], "filename": "COCO_val2014_000000391895.jpg", "imgid": 0, "split": "test", "sentences": [{"tokens": ["a", "man", "with", "a", "red", "helmet", "on    ", "a", "small", "moped", "on", "a", "dirt", "road"], "raw": "A man with a red helmet on a small moped on a dirt road. ", "imgid": 0, "sentid": 770337}, {"tokens": ["man", "riding", "a", "motor", "bike", "on", "a", "dirt", "road", "on    ", "the", "countryside"], "raw": "Man riding a motor bike on a dirt road on the countryside.", "imgid": 0, "sentid": 771687}, {"tokens": ["a", "man", "riding", "on", "the", "back", "of", "a", "motorcycle"], "raw": "A man riding on the     back of a motorcycle.", "imgid": 0, "sentid": 772707}, {"tokens": ["a", "dirt", "path", "with", "a", "young", "person", "on", "a", "motor", "bike", "rests", "to", "the", "foreground", "of", "a", "verdant", "area", "with", "a", "bridg    e", "and", "a", "background", "of", "cloud", "wreathed", "mountains"], "raw": "A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ", "i    mgid": 0, "sentid": 776154}, {"tokens": ["a", "man", "in", "a", "red", "shirt", "and", "a", "red", "hat", "is", "on", "a", "motorcycle", "on", "a", "hill", "side"], "raw": "A man in a red shirt and a red hat is on a motorcycle on a hi    ll side.", "imgid": 0, "sentid": 781998}], "cocoid": 391895} 

def split_id2path(args): 
    
    # construct ID: {wav_path, transcript_path, tree_path, alignment_path}
    id2path = defaultdict(list)
    total, no_align, too_long = 0, 0, 0
    cnt_statistics = [total, no_align, too_long]
    with open(args.missing_file, 'w') as f: 
        cnt_statistics = construct_id2path(args.input_train_file, args, f, cnt_statistics, id2path)
        cnt_statistics = construct_id2path(args.input_val_file, args, f, cnt_statistics, id2path)

    print('There are %d total utterances. %d has no alignments. %d exceeds 15 seconds' % (cnt_statistics[0], cnt_statistics[1], cnt_statistics[2]))

def construct_id2path(input_json_file, args, f, cnt_statistics, id2path = {}): 
    data = json.load(open(input_json_file))
    
    for pair in tqdm(data['data']):
        img_key = pair['image'].split('/')[1]
        ID = img_key
        captions = pair['captions']
        for caption in captions: # each image pairs with 5 captions
            cnt_statistics[0] += 1
            wav_file = os.path.join(args.data_directory, args.new_data_directory_name, caption['speaker'], caption['uttid'] + '.wav')
            alignment_file = wav_file.replace('/wavs-speaker/', '/wavs-speaker-aligned/').replace('.wav', '.TextGrid')
            
            assert exists(wav_file), print(wav_file)
            
            if not exists(alignment_file):
                print(wav_file, 'does not have alignment. Skipped')
                f.write('speaker %s file %s does not have alignment\n' % (caption['speaker'], wav_file))
                cnt_statistics[1] += 1
                print('There are %d total utterances. %d has no alignments. %d exceeds 15 seconds' % (cnt_statistics[0], cnt_statistics[1], cnt_statistics[2]))
            elif skip_long_utterance(wav_file): 
                print(wav_file, 'exceeds 15 seconds. Skipped')
                f.write('speaker %s file %s exceeds 15 seconds.\n' % (caption['speaker'], wav_file))
                cnt_statistics[2] += 1 
                print('There are %d total utterances. %d has no alignments. %d exceeds 15 seconds' % (cnt_statistics[0], cnt_statistics[1], cnt_statistics[2]))

    return cnt_statistics

def skip_long_utterance(wav_file, cut_off_len = 15): 
    if os.path.islink(wav_file): 
        wav_file = os.readlink(wav_file)

    with contextlib.closing(wave.open(wav_file, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    if duration >= cut_off_len: 
        return True 
    else: 
        return False

         
if __name__ == '__main__':
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-val-file', '-v', type=str)
    parser.add_argument('--input-train-file', '-t', type=str)
    parser.add_argument('--data-directory', '-d', type=str)
    parser.add_argument('--new-data-directory-name', '-n', type=str)
    parser.add_argument('--split-file', '-s', type=str)
    parser.add_argument('--output-summary-json', '-o', type=str)
    parser.add_argument('--missing-file', '-m', type=str)
    args = parser.parse_args()

    split_id2path(args)
   
