import numpy as np 
import os.path as osp

feature='logmelspec' # 'hubert'
for seg_type in ['phn', 'word']: 
    for data_split in ['val', 'test', 'train']:
        seg_list_pth = osp.join('data/SpokenCOCO/Freda-formatting/', data_split + '_segment-' + feature + '_' + seg_type + '_list-83k-5k-5k.npy')
        seg_lists = np.load(seg_list_pth, allow_pickle=True)[0]

        total_seg_len = 0
        total_seg = 0
        max_seg_len = -1
        for i in range(len(seg_lists)): 
            seg_list = seg_lists[i] 
            seg2len = [round(z)-round(y) for (_,y,z) in seg_list]

            total_seg_len += sum(seg2len) 
            total_seg += len(seg2len)
            max_seg_len = max(max_seg_len, max(seg2len))

        print('data_split is %s' % data_split)
        print('feature is %s' % feature)
        print('seg_type is %s' % seg_type)
        print('average segment_len is %f' % (total_seg_len / total_seg))
        print('max segment_len is %f' % max_seg_len)
        print('\n')
