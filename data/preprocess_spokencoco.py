import json 
import argparse 
from tqdm import tqdm
import os

def export_transcript(args):
    data = json.load(open(args.input_file))
    # {"image": "val2014/COCO_val2014_000000325114.jpg", "captions": [{"text": "A URINAL IN A PUBLIC RESTROOM NEAR A WOODEN TABLE", "speaker": "m071506418gb9vo0w5xq3", "uttid": "m071506418gb9vo0w5xq3-3LUY3GC63Z0R9PYEETJGN5HO4UEP7B    _325114_629297", "wav": "wavs/val/0/m071506418gb9vo0w5xq3-3LUY3GC63Z0R9PYEETJGN5HO4UEP7B_325114_629297.wav", "parse_tree": "( ( a urinal ) ( in ( ( a public restroom ) ( near ( a wooden table ) ) ) ) )"}, {"text": "A WHITE URINAL WHIT    E TILES A RED TABLE WITH A DISH", "speaker": "m221bzbcdh4ro4", "uttid": "m221bzbcdh4ro4-3ZQIG0FLQEGJ4OWB8D0RLD5NKVLWVB_325114_628718", "wav": "wavs/val/0/m221bzbcdh4ro4-3ZQIG0FLQEGJ4OWB8D0RLD5NKVLWVB_325114_628718.wav", "parse_tree":     "( ( a white urinal white ) ( tiles ( a red table ) ( with ( a dish ) ) ) )"}, {"text": "A TABLE WITH A PLATE OF FOOD AND A URINAL BASIN", "speaker": "m2zco7272xa5e0", "uttid": "m2zco7272xa5e0-3RANCT1ZVFHR36908WUQ2DQJVTZBU1_325114_615    377", "wav": "wavs/val/0/m2zco7272xa5e0-3RANCT1ZVFHR36908WUQ2DQJVTZBU1_325114_615377.wav", "parse_tree": "( ( a table ) ( with ( ( ( a plate ) ( of food ) ) and ( a urinal basin ) ) ) )"}, {"text": "A LITTLE RED TABKE WITH A PLATE ON     IT IN THE BATHROOM", "speaker": "m2saqi3zjrms1y", "uttid": "m2saqi3zjrms1y-3F0BG9B9MPNLI3QF5GFZ0WA089EY7V_325114_630704", "wav": "wavs/val/0/m2saqi3zjrms1y-3F0BG9B9MPNLI3QF5GFZ0WA089EY7V_325114_630704.wav", "parse_tree": "( ( a little     red tabke ) ( with ( ( a plate ) ( on it ) ( in ( the bathroom ) ) ) ) )"}, {"text": "A URINAL NEXT TO A RED TABLE WITH A PLATE SITTING ON THE TABLE", "speaker": "ma7soiuo8jbku", "uttid": "ma7soiuo8jbku-3LOTDFNYA7ZU8RAL8YVN3R21Z9QFWI    _325114_622604", "wav": "wavs/val/0/ma7soiuo8jbku-3LOTDFNYA7ZU8RAL8YVN3R21Z9QFWI_325114_622604.wav", "parse_tree": "( ( a urinal ) ( next ( to ( ( a red table ) ( with ( ( a plate ) ( sitting ( on ( the table ) ) ) ) ) ) ) ) )"}]}
    
    for pair in tqdm(data['data']):
        img = pair['image']
        captions = pair['captions']
        for caption in captions: # each image pairs with 5 captions
            transcript = caption['text'].lower()
            target_file = os.path.join(args.data_directory, caption['wav'].replace('.wav', '.txt'))
            f = open(target_file, 'w') 
            f.write(transcript)
            f.close()

if __name__ == '__main__':
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', '-i', type=str)
    parser.add_argument('--data-directory', '-d', type=str)
    args = parser.parse_args()

    export_transcript(args) 
   
