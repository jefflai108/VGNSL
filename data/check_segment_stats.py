import numpy as np 
import os.path as osp

jason_feature = None 
jason_feature = 'disc-81_spokencoco_preFeats_max_0.7_9_clsAttn'
jason_feature = 'disc-81_spokencoco_preFeats_weightedmean_0.8_9_clsAttn'
jason_feature = 'model3_spokencoco_preFeats_weightedmean_0.8_7_clsAttn'
jason_feature = 'disc-26_spokencoco_preFeats_weightedmean_0.8_7_clsAttn'

feature='logmelspec' # 'hubert'
for seg_type in ['phn', 'word']: 
    for data_split in ['val', 'test', 'train']:
        if jason_feature is None:
            seg_list_pth = osp.join('data/SpokenCOCO/Freda-formatting/', data_split + '_segment-' + feature + '_' + seg_type + '_list-83k-5k-5k.npy')
        else: 
            seg_list_pth = osp.join('data/SpokenCOCO/Freda-formatting/', data_split + '-' + jason_feature + '-pred_word_list-83k-5k-5k.npy')
        seg_lists = np.load(seg_list_pth, allow_pickle=True)[0]

        total_seg_len = 0
        total_seg = 0
        max_seg_len = -1
        for i in range(len(seg_lists)): 
            seg_list = seg_lists[i] 
            seg2len = [round(z)-round(y) for (_,y,z) in seg_list]
            if jason_feature is not None:
                seg2len = [x/0.02 for x in seg2len] # for hubert 

            total_seg_len += sum(seg2len) 
            total_seg += len(seg2len)
            max_seg_len = max(max_seg_len, max(seg2len))

        print('data_split is %s' % data_split)
        print('feature is %s' % feature)
        print('seg_type is %s' % seg_type)
        print('average segment_len is %f' % (total_seg_len / total_seg))
        print('max segment_len is %f' % max_seg_len)
        print('\n')
