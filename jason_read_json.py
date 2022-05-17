import json 

with open('summary_jason_file', 'r') as f: 
    data_summary = json.read(f) 
data_summary = data_summary['val']

image_key_list = []
for img_key, caption_list in data_summary.items(): 
    image_key_list.append(img_key)
   
def __deduplicate__(captions_list):
    # ensure image:captions == 1:5
    if len(captions_list) > 5: 
        captions_list = captions_list[:5]
    while len(captions_list) < 5: # duplicate 
        captions_list.append(captions_list[-1])
    assert len(captions_list) == 5

    return captions_list

np_file_idx = 0 
for img_key in image_key_list: 
    caption_list = __deduplicate__(data_summary[img_key])
    for caption in caption_list: 
        wav_file = caption[0]
        
        print('correspondance: %d %s' % (np_file_idx, wav_file))
        np_file_idx += 1
