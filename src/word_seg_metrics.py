import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

from module import AttentivePooling
from utils import l2norm

def word_seg_F1(pred_seg_vecs, gt_seg_vecs): 

    num_of_phn_seg_tolerance = 0

    precision_counter = 0
    recall_counter = 0
    pred_counter = 0
    gt_counter = 0
    for (y, yhat) in zip(gt_seg_vecs, pred_seg_vecs):
        for yhat_i in yhat:
            min_dist = np.abs(y - yhat_i).min()
            precision_counter += (min_dist <= num_of_phn_seg_tolerance)
        for y_i in y:
            min_dist = np.abs(yhat - y_i).min()
            recall_counter += (min_dist <= num_of_phn_seg_tolerance)
        pred_counter += len(yhat)
        gt_counter += len(y)

    p, r, f1, rval = get_metrics(precision_counter,
                                 recall_counter,
                                 pred_counter,
                                 gt_counter)
    print('f1 is %f and rval is %f' % (f1, rval))

def get_metrics(precision_counter, recall_counter, pred_counter, gt_counter):
    EPS = 1e-7
    
    precision = precision_counter / (pred_counter + 1e-5)
    recall = recall_counter / (gt_counter + 1e-5)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-5)
    
    os = recall / (precision + EPS) - 1
    r1 = np.sqrt((1 - recall) ** 2 + os ** 2)
    r2 = (-os + recall - 1) / (np.sqrt(2))
    rval = 1 - (np.abs(r1) + np.abs(r2)) / 2

    return precision, recall, f1, rval

if __name__ == '__main__': 
    
    # word segmentation results 
    pred_seg_vecs = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    gt_seg_vecs = torch.tensor([1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1])
    pred_seg_vecs = (pred_seg_vecs == 1).nonzero(as_tuple=True)[0]
    gt_seg_vecs = (gt_seg_vecs == 1).nonzero(as_tuple=True)[0]
    print(gt_seg_vecs)
    word_seg_F1(pred_seg_vecs.repeat(10, 1), gt_seg_vecs.repeat(10, 1))


