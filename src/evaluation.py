import os
import random
import sys
import pickle
import regex
import time
from tqdm import tqdm
from collections import OrderedDict

import numpy
import numpy as np
import torch
from nltk import Tree
from IPython import embed

from model import VGNSL
from vocab import Vocabulary
from data import get_eval_loader
from utils import generate_tree, clean_tree

sys.path.insert(-1, os.path.join(sys.path[0], '../analysis'))
from constituent_recall import constituent_recall
from ex_sparseval import corpus_f1 as ex_sparseval_f1

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s


def encode_data(data_path, basename, model, data_loader, log_step=10, logging=print, vocab=None, stage='dev', speech_hdf5=False, 
                phn_force_align=False, diffbound_gtword=False,
                unsup_word_discovery_feats=None, unsup_word_discovery_feat_type=None):
    """Encode all images and captions loadable by `data_loader`
    """
    if phn_force_align: 
        ground_truth = [line.strip().lower() for line in open(
            os.path.join(data_path, f'val_phn-level-ground-truth-{basename}.txt'))]
        if diffbound_gtword: 
            ground_truth = [line.strip() for line in open(
                os.path.join(data_path, f'val_word-level-ground-truth-{basename}.txt'))]
    else: 
        ground_truth = [line.strip() for line in open(
            os.path.join(data_path, f'val_word-level-ground-truth-{basename}.txt'))]
    if unsup_word_discovery_feats: # load alignments
        unsup_discovered_word_alignments = np.load(os.path.join(data_path, f'val-{unsup_word_discovery_feats}-{unsup_word_discovery_feat_type}_alignment_via_max_weight_matching-{basename}.npy'), allow_pickle=True)[0]
        unsup_discovered_word_alignments = list(unsup_discovered_word_alignments.values())

    batch_time = AverageMeter()
    val_logger = LogCollector()
    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    logged = False
    trees = list()
    for i, (images, captions, audios, audio_masks, lengths, ids) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger
        lengths = torch.Tensor(lengths).long()
        if torch.cuda.is_available():
            lengths = lengths.cuda()

        # compute the embeddings
        model_output = model.forward_emb(images, audios, lengths, volatile=True, speech_hdf5=speech_hdf5, audio_masks=audio_masks) # feed in audios instead of captions
        img_emb, cap_span_features, left_span_features, right_span_features, word_embs, tree_indices, all_probs, \
        span_bounds = model_output[:8]

        # output sampled trees
        if (not logged) or (stage == 'test'):
            logged = True
            if stage == 'dev':
                sample_num = 5
            for j in range(sample_num):
                logging(generate_tree(captions, tree_indices, j, vocab)) # visualize generated tree on top of captions

        # store trees
        candidate_trees = list()
        for j in range(len(ids)):
            candidate_trees.append(generate_tree(captions, tree_indices, j, vocab))

        # re-order trees
        appended_trees = ['' for _ in range(len(ids))] # mini-batch produced trees
        batched_ground_truth = ['' for _ in range(len(ids))] # mini-batch gruond_truth trees
        for j in range(len(ids)):
            appended_trees[ids[j] - min(ids)] = clean_tree(candidate_trees[j])
            batched_ground_truth[ids[j] - min(ids)] = ground_truth[ids[j]]
        trees.extend(appended_trees)
        
        # cap emb
        cap_emb = torch.cat([cap_span_features[l-2][i].reshape(1, -1) for i, l in enumerate(lengths)], dim=0)

        # initialize the numpy arrays given the size of the embeddings
        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))

        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids] = cap_emb.data.cpu().numpy().copy()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                        i, len(data_loader), batch_time=batch_time,
                        e_log=str(model.logger)))
        del images, captions, img_emb, cap_emb, audios, audio_masks

    if unsup_word_discovery_feats:
        trees, ground_truth, unsup_discovered_word_alignments = _cleanup_tree(trees, ground_truth, unsup_discovered_word_alignments)
        f1 = ex_sparseval_f1(ground_truth, trees, unsup_discovered_word_alignments, is_baretree=True) # careful of the ordering: gold_trees --> pred_trees
    else: # normal corpus f1
        f1, _, _ =  f1_score(trees, ground_truth)
    logging(f'validation tree f1 score is {f1}')

    return img_embs, cap_embs


def i2t(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = int(images.shape[0] / 5)
        # print(npts)
    index_list = []

    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[5 * index].reshape(1, images.shape[1])

        # Compute scores
        d = numpy.dot(im, captions.T).flatten()
        inds = numpy.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = int(images.shape[0] / 5)
        # print(npts)
    ims = numpy.array([images[i] for i in range(0, len(images), 5)])

    ranks = numpy.zeros(5 * npts)
    top1 = numpy.zeros(5 * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[5 * index:5 * index + 5]

        # compute scores
        d = numpy.dot(queries, ims.T)
        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            ranks[5 * index + i] = numpy.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    # compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def test_trees(data_path, model_path, vocab_path, basename, data_split='test',
               visual_tree=False, visual_samples=10,
               export_tree=False, export_tree_path=None, 
               constituent_recall=False, duration_based_alignment=False, 
               test_time_oracle_segmentation=False, mbr_path=None, 
               right_branching=False, left_branching=False, random_branching=False):
    """ use the trained model to generate parse trees for text """
    # load model and options
    checkpoint = torch.load(model_path, map_location='cpu')
    opt = checkpoint['opt']

    # load vocabulary used by the model
    try:
        vocab = pickle.load(open(vocab_path, 'rb'))
    except:
        import pickle5
        vocab = pickle5.load(open(vocab_path, 'rb'))
    opt.vocab_size = len(vocab)

    # construct model
    model = VGNSL(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])
    if hasattr(opt, 'logmelspec_cmvn'):
        use_cmvn = opt.logmelspec_cmvn
    elif hasattr(opt, 'feature_cmvn'): 
        use_cmvn = opt.feature_cmvn
    if hasattr(opt, 'discretized_phone'): 
        use_discretized_phone = opt.discretized_phone
    else: use_discretized_phone = False
    if hasattr(opt, 'discretized_word'): 
        use_discretized_word = opt.discretized_word
    else: use_discretized_word = False
    if hasattr(opt, 'km_clusters'): 
        km_clusters = opt.km_clusters
    else: km_clusters = 0
    if hasattr(opt, 'phn_force_align'): 
        phn_force_align = opt.phn_force_align
    else: phn_force_align = False 
    if hasattr(opt, 'diffbound_gtword'): 
        diffbound_gtword = opt.diffbound_gtword
    else: diffbound_gtword = False
    if hasattr(opt, 'dino_feature'): 
        dino_feature = opt.dino_feature
    else: dino_feature = None
    if hasattr(opt, 'use_seg_feats_for_unsup_word_discovery'): 
        use_seg_feats_for_unsup_word_discovery = opt.use_seg_feats_for_unsup_word_discovery
    else: use_seg_feats_for_unsup_word_discovery = False 
    if hasattr(opt, 'unsup_word_discovery_feat_type'): 
        unsup_word_discovery_feat_type = opt.unsup_word_discovery_feat_type
    else: unsup_word_discovery_feat_type = None 
    if hasattr(opt, 'unsup_word_discovery_feats'): 
        unsup_word_discovery_feats = opt.unsup_word_discovery_feats
    else: unsup_word_discovery_feats = None
    if hasattr(opt, 'uniform_word_force_align'): 
        uniform_word_force_align = opt.uniform_word_force_align
    else: uniform_word_force_align = False 
    
    if visual_tree: 
        eval_batch_size = 1 
    elif export_tree: # smaller batch size to avoid mem error
        eval_batch_size = 128
    else: eval_batch_size = opt.batch_size

    data_loader = get_eval_loader(
        data_path, data_split, vocab, basename, eval_batch_size, 1,
        feature=opt.feature, load_img=False, img_dim=opt.img_dim, utt_cmvn=use_cmvn, speech_hdf5=opt.speech_hdf5, 
        discretized_phone=use_discretized_phone, discretized_word=use_discretized_word, km_clusters=km_clusters, 
        phn_force_align=phn_force_align, diffbound_gtword=diffbound_gtword, dino_feature=dino_feature, 
        unsup_word_discovery_feats=unsup_word_discovery_feats, unsup_word_discovery_feat_type=unsup_word_discovery_feat_type, 
        use_seg_feats_for_unsup_word_discovery=use_seg_feats_for_unsup_word_discovery, uniform_word_force_align=uniform_word_force_align, 
        test_time_oracle_segmentation=test_time_oracle_segmentation
    )

    if phn_force_align: # phn-level alignment 
        ground_truth = [line.strip().lower() for line in open(
            os.path.join(data_path, f'{data_split}_phn-level-ground-truth-{basename}.txt'))]
        all_captions = [line.strip() for line in open(
            os.path.join(data_path, f'{data_split}_phn_caps-{basename}.txt'))]

        if diffbound_gtword: # differential boundary setup: use word-level text captions
            ground_truth = [line.strip() for line in open(
                os.path.join(data_path, f'{data_split}_word-level-ground-truth-{basename}.txt'))]
            all_captions = [line.strip() for line in open(
                os.path.join(data_path, f'{data_split}_caps-{basename}.txt'))]
    else: # word-level alignment 
        ground_truth = [line.strip() for line in open(
            os.path.join(data_path, f'{data_split}_word-level-ground-truth-{basename}.txt'))]
        all_captions = [line.strip() for line in open(
            os.path.join(data_path, f'{data_split}_caps-{basename}.txt'))]
    if unsup_word_discovery_feats: # load alignments
        if duration_based_alignment: # duration-based alignment 
            #print('loading duration-based alignment!')
            unsup_discovered_word_alignments = np.load(os.path.join(data_path, f'{data_split}-{unsup_word_discovery_feats}-{unsup_word_discovery_feat_type}_alignment_via_max_weight_duration_matching-{basename}.npy'), allow_pickle=True)[0]
        else: # default is l1-distance based alignment
            #print('loading l1-based alignment!')
            unsup_discovered_word_alignments = np.load(os.path.join(data_path, f'{data_split}-{unsup_word_discovery_feats}-{unsup_word_discovery_feat_type}_alignment_via_max_weight_matching-{basename}.npy'), allow_pickle=True)[0]
        unsup_discovered_word_alignments = list(unsup_discovered_word_alignments.values())

    if visual_tree: 
        ground_truth = ground_truth[:visual_samples]
        all_captions = all_captions[:visual_samples]
    if export_tree: 
        export_tree_writer = open(export_tree_path, 'w')
    cap_embs = None
    logged = False
    trees = list()

    for i, (images, captions, audios, audio_masks, lengths, ids) in enumerate(data_loader):
        if visual_tree and i == visual_samples: 
            break 
        # make sure val logger is used
        model.logger = print
        lengths = torch.Tensor(lengths).long()
        if torch.cuda.is_available():
            lengths = lengths.cuda()

        # compute the embeddings
        model_output = model.forward_emb(images, audios, lengths, volatile=True, speech_hdf5=opt.speech_hdf5, audio_masks=audio_masks) # feed in audios instead of captions
        img_emb, cap_span_features, left_span_features, right_span_features, word_embs, tree_indices, all_probs, \
        span_bounds = model_output[:8]

        candidate_trees = list()
        for j in range(len(ids)):
            candidate_tree = generate_tree(captions, tree_indices, j, vocab)
            candidate_trees.append(candidate_tree)

        appended_trees = ['' for _ in range(len(ids))]
        for j in range(len(ids)):
            appended_trees[ids[j] - min(ids)] = clean_tree(candidate_trees[j])
        trees.extend(appended_trees)
        
        # process inferred tree by mini-batch due to memory error 
        if export_tree:
            #for id in ids: print(id)
            start_id, end_id = ids[0], ids[-1]
            for tree in trees: 
                export_tree_writer.write('%s\n' % tree)
            f1, _, _ = f1_score(trees, ground_truth[start_id:end_id+1], all_captions[start_id:end_id+1])
            print(f'Current batch tree f1 is {f1}')
            del trees, candidate_trees
            torch.cuda.empty_cache()
            trees = list() # refresh after mini-batch 

        #cap_emb = torch.cat([cap_span_features[l-2][i].reshape(1, -1) for i, l in enumerate(lengths)], dim=0)
        del images, captions, img_emb, audios, audio_masks, lengths, ids, model_output, \
            cap_span_features, left_span_features, right_span_features, word_embs, tree_indices, all_probs, span_bounds

    if export_tree: 
        export_tree_writer.close()
        exit()

    if mbr_path: # pre-store predicted trees for all checkpoints. They will be used later for MBR model selection.
        print(f'writing predicted trees to {mbr_path}')
        with open(mbr_path, 'w') as f: 
            for tree in trees: 
                f.write('%s\n' % tree) 

    if unsup_word_discovery_feats and not test_time_oracle_segmentation: # if test_time_oracle_segmentation, use normal F1 score. 
        trees, ground_truth, unsup_discovered_word_alignments = _cleanup_tree(trees, ground_truth, unsup_discovered_word_alignments)
        f1 = ex_sparseval_f1(ground_truth, trees, unsup_discovered_word_alignments, is_baretree=True) # careful of the ordering: gold_trees --> pred_trees
       
        ## tree baselines for unsup_word_discovery_feats
        #right_branching=True
        #left_branching=True 
        #random_branching=True 
        if right_branching: 
            print(f'Right branching')
            right_trees = [right_branching_algorithm(x) for x in trees]
            rf1 = ex_sparseval_f1(ground_truth, right_trees, unsup_discovered_word_alignments, is_baretree=True) # careful of the ordering: gold_trees --> pred_trees
            print(f'\tCorpus sparseval F1: {rf1:.3f}')
        if left_branching: 
            print(f'Left branching')
            left_trees = [left_branching_algorithm(x) for x in trees]
            lf1 = ex_sparseval_f1(ground_truth, left_trees, unsup_discovered_word_alignments, is_baretree=True) # careful of the ordering: gold_trees --> pred_trees
            print(f'\tCorpus sparseval F1: {lf1:.3f}')
        if random_branching:
            print(f'Random branching')
            c_rf1 = 0
            for seed in range(5):
                random_trees = [random_branching_algorithm(x, seed) for x in trees]
                rf1 = ex_sparseval_f1(ground_truth, random_trees, unsup_discovered_word_alignments, is_baretree=True) # careful of the ordering: gold_trees --> pred_trees
                c_rf1 += rf1
            c_rf1 /= 5
            print(f'\tCorpus sparseval F1: {c_rf1:.3f}')
    else: # normal corpus f1
        f1, _, _ = f1_score(trees, ground_truth, all_captions, visual_tree, constituent_recall)

    return f1

def left_branching_algorithm(st):
    words = st.replace('(', '').replace(')', '').split()
    if len(words) == 1:
        return (f'( {words[0]} )')
    else:
        current_st = f'( {words[0]} {words[1]} )'
        for item in words[2:]:
            current_st = f'( {current_st} {item} )'
        return current_st

def right_branching_algorithm(st):
    words = st.replace('(', '').replace(')', '').split()
    if len(words) == 1:
        return (f'( {words[0]} )')
    else:
        current_st = f'( {words[-2]} {words[-1]} )'
        for item in words[2:]:
            current_st = f'( {item} {current_st} )'
        return current_st

def random_branching_algorithm(st, seed=0):
    random.seed(seed)
    words = st.replace('(', '').replace(')', '').split()
    while len(words) > 1:
        position = random.randint(0, len(words) - 2)
        item = f'( {words[position]} {words[position+1]} )'
        words = words[:position] + [item] + words[position+2:]
    return words[0]

def viz_tree(bare_tree):
    nt_tree = bare_tree.replace('(', '(NT').replace(' ', '  ')
    nt_tree = regex.sub(r' ([^ \(\)]+) ', r' (PT \1) ', nt_tree)
    nltk_tree = Tree.fromstring(nt_tree)
    nltk_tree.pretty_print()

def _retrieve_text_from_tree(tree): 
    return ' '.join(tree.replace('(', '').replace(')', '').split())

def _cleanup_tree(orig_produced_trees, orig_gold_trees, unsup_discovered_word_alignments=None):
    # remove word-level mismatch (from pre-processing)
    # by keeping track of the indices 
    indices_to_remove = []
    for i in range(len(orig_gold_trees)):
        if orig_gold_trees[i] in ['MISMATCH', 'mismatch']:
            indices_to_remove.append(i)
    orig_gold_trees = [orig_gold_tree for i, orig_gold_tree in enumerate(orig_gold_trees) if i not in indices_to_remove]
    orig_produced_trees = [orig_produced_tree for i, orig_produced_tree in enumerate(orig_produced_trees) if i not in indices_to_remove]
    if unsup_discovered_word_alignments: 
        unsup_discovered_word_alignments = [alignment for i, alignment in enumerate(unsup_discovered_word_alignments) if i not in indices_to_remove]

    return orig_produced_trees, orig_gold_trees, unsup_discovered_word_alignments

def f1_score(orig_produced_trees, orig_gold_trees, captions=None, visual_tree=False, constituent_recall_analysis=False):
    orig_produced_trees, orig_gold_trees, _ = _cleanup_tree(orig_produced_trees, orig_gold_trees)

    # double-check underlying word/phn sequence match 
    orig_gold_trees_text = [_retrieve_text_from_tree(orig_gold_tree) for orig_gold_tree in orig_gold_trees]
    orig_produced_trees_text = [_retrieve_text_from_tree(orig_produced_tree) for orig_produced_tree in orig_produced_trees]
    assert orig_gold_trees_text == orig_produced_trees_text # underlying words/phones should match. 

    # constituency recall analysis 
    if constituent_recall_analysis:
        recall = constituent_recall(orig_gold_trees_text, orig_produced_trees)
        print(recall)

    # compute f1 score over spans 
    gold_trees = list(map(lambda tree: extract_spans(tree), orig_gold_trees))
    produced_trees = list(map(lambda tree: extract_spans(tree), orig_produced_trees))
    assert len(produced_trees) == len(gold_trees), print(len(produced_trees), len(gold_trees))
    precision_cnt, precision_denom, recall_cnt, recall_denom = 0, 0, 0, 0
    for i, item in enumerate(produced_trees):
        if visual_tree: 
            print('\ntarget captions:')
            print(f'{captions[i]}')
            print('\ngold tree:\n')
            viz_tree(orig_gold_trees[i])
            print('\ninduced tree:\n')
            viz_tree(orig_produced_trees[i])
        pc, pd, rc, rd = extract_statistics(gold_trees[i], item)
        precision_cnt += pc
        precision_denom += pd
        recall_cnt += rc
        recall_denom += rd
    precision = float(precision_cnt) / precision_denom * 100.0
    recall = float(recall_cnt) / recall_denom * 100.0
    f1 = 2 * precision * recall / (precision + recall)
       
    return f1, precision, recall


def extract_spans(tree):
    answer = list()
    stack = list()
    items = tree.split()
    curr_index = 0
    for item in items:
        if item == ')':
            pos = -1
            right_margin = stack[pos][1]
            left_margin = None
            while stack[pos] != '(':
                left_margin = stack[pos][0]
                pos -= 1
            assert left_margin is not None
            assert right_margin is not None
            stack = stack[:pos] + [(left_margin, right_margin)]
            answer.append((left_margin, right_margin))
        elif item == '(':
            stack.append(item)
        else:
            stack.append((curr_index, curr_index))
            curr_index += 1
    return answer


def extract_statistics(gold_tree_spans, produced_tree_spans):
    gold_tree_spans = set(gold_tree_spans)
    produced_tree_spans = set(produced_tree_spans)
    precision_cnt = sum(list(map(lambda span: 1.0 if span in gold_tree_spans else 0.0, produced_tree_spans)))
    recall_cnt = sum(list(map(lambda span: 1.0 if span in produced_tree_spans else 0.0, gold_tree_spans)))
    precision_denom = len(produced_tree_spans)
    recall_denom = len(gold_tree_spans)
    return precision_cnt, precision_denom, recall_cnt, recall_denom
