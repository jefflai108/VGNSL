import numpy as np

def find_boundary_matches(gt, pred, tolerance):
    """
    gt: list of ground truth boundaries
        e.g. [0.07, 0.12, 0.33, 0.45, 1.10]
    pred: list of predicted boundaries
        e.g. same format as gt
    tolerance: 0.02
    """
    gt_pointer = 0
    pred_pointer = 0
    gt_len = len(gt)
    pred_len = len(pred)
    match_pred = 0
    match_gt = 0
    while gt_pointer < gt_len and pred_pointer < pred_len:
        if np.abs(gt[gt_pointer] - pred[pred_pointer]) <= tolerance:
            match_gt += 1
            match_pred += 1
            gt_pointer += 1
            pred_pointer += 1
        elif gt[gt_pointer] > pred[pred_pointer]:
            pred_pointer += 1
        else:
            gt_pointer += 1
    # # below is method used in Kreuk et al, which I didn't used for SpokenCOCO
    # for pred_i in pred:
    #     min_dist = np.abs(gt - pred_i).min()
    #     match_pred += (min_dist <= tolerance)
    # for y_i in gt:
    #     min_dist = np.abs(pred - y_i).min()
    #     match_gt += (min_dist <= tolerance)
    return match_gt, match_pred, gt_len, pred_len

def main(all_pred_boundaries, all_gt_boundaries):
    match_gt_count = 0
    match_pred_count = 0
    gt_b_len = 0
    pred_b_len = 0
    tolerance = 0.02 # in seconds, this requires that boundaries are also in seconds
    for pred_boundaries, gt_boundaries in zip(all_pred_boundaries, all_gt_boundaries): # loop through SpokenCOCO dataset
        a, b, c, d = find_boundary_matches(gt_boundaries[1:-1], pred_boundaries[1:-1], tolerance) # exclude the first and last boundaries, you don't have to, but I did this in the paper
        match_gt_count += a
        match_pred_count += b
        gt_b_len += c
        pred_b_len += d

    b_prec = match_pred_count / pred_b_len
    b_recall = match_gt_count / gt_b_len
    b_f1 = 2*b_prec*b_recall / (b_prec+b_recall)
    b_os = b_recall / b_prec - 1.
    b_r1 = np.sqrt((1-b_recall)**2 + b_os**2)
    b_r2 = (-b_os + b_recall - 1) / np.sqrt(2)
    b_r_val = 1. - (np.abs(b_r1) + np.abs(b_r2))/2.
    print("precision: ", b_prec)
    print("recall: ", b_recall)
    print("f1: ", b_f1)
    print("oversegmentation: ", b_os)
    print("R-value: ", b_r_val)

if __name__ == '__main__': 
    
    main()
    exit()

    # word segmentation results 
    pred_seg_vecs = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    gt_seg_vecs = torch.tensor([1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1])
    pred_seg_vecs = (pred_seg_vecs == 1).nonzero(as_tuple=True)[0]
    gt_seg_vecs = (gt_seg_vecs == 1).nonzero(as_tuple=True)[0]
    print(gt_seg_vecs)
    word_seg_F1(pred_seg_vecs.repeat(10, 1), gt_seg_vecs.repeat(10, 1))


