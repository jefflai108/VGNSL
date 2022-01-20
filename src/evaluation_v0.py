import os
import pickle

import numpy
from data import get_eval_loader
import time
import numpy as np
from vocab import Vocabulary 
import torch
from model import VGNSL
from collections import OrderedDict

from utils import generate_tree, clean_tree
from IPython import embed

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


def encode_data(model, data_loader, log_step=10, logging=print, vocab=None, stage='dev'):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()
    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    logged = False
    for i, (images, captions, lengths, ids, keys) in enumerate(data_loader):
        #print(ids)
        # make sure val logger is used
        model.logger = val_logger
        lengths = torch.Tensor(lengths).long()
        if torch.cuda.is_available():
            lengths = lengths.cuda()

        # compute the embeddings
        model_output = model.forward_emb(images, captions, lengths, volatile=True)
        img_emb, cap_span_features, left_span_features, right_span_features, word_embs, tree_indices, all_probs, \
        span_bounds = model_output[:8]

        # output sampled trees
        if (not logged) or (stage == 'test'):
            logged = True
            if stage == 'dev':
                sample_num = 1 # set it to 1 since batch_size is hard coded to 1
            for j in range(sample_num):
                logging(generate_tree(captions, tree_indices, j, vocab))

        cap_emb = torch.cat([cap_span_features[l-2][i].reshape(1, -1) for i, l in enumerate(lengths)], dim=0)

        # initialize the numpy arrays given the size of the embeddings
        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))
            key_list = []

        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids] = cap_emb.data.cpu().numpy().copy()
        key_list.extend(keys)
        
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
        del images, captions, keys

    return img_embs, cap_embs, np.array(key_list)

def t2i(images, captions, keys, npts=None, measure='cosine', return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    ranks = numpy.zeros(len(keys))
    top1  = numpy.zeros(len(keys))

    # create image search space 
    uniq_keys = dict.fromkeys(keys) # preserve original order
    selected_idxs = []
    for uniq_key in uniq_keys: 
        selected_idx = np.where(keys == uniq_key)[0]
        selected_idxs.append(selected_idx[0]) # only need to take the first as representative 

        # double-check 
        # ensure for each uniq_key, the image embeddings are the same 
        _selected_img = images[selected_idx]
        if len(_selected_img) > 1:
            assert sum(_selected_img[0] - _selected_img[-1]) == 0
    #print(selected_idxs)
    unique_images = images[np.array(selected_idxs)]
    #print(unique_images.shape)

    counter = 0
    for query_idx, query_key in enumerate(uniq_keys): 
        # Get query captions 
        selected_idx = np.where(keys == query_key)[0]
        selected_caption = captions[selected_idx]

        # compute scores (selected_caption v.s. unique_images)
        #print(selected_caption.shape) # 5, 512
        #print(unique_images.shape) # 30, 512
        similarity_score = np.dot(selected_caption, unique_images.T) # 5, 30
        similarity_score_inds = np.zeros(similarity_score.shape)
        #print(similarity_score.shape)
        
        for i in range(len(similarity_score)): # iterate over captions under the query_key
            similarity_score_inds[i] = np.argsort(similarity_score[i])[::-1]
            top1[counter] = similarity_score_inds[i][0] # highest similarity for the given caption
            ranks[counter] = np.where(similarity_score_inds[i] == query_idx)[0][0]
            counter += 1

    # compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)

def i2t(images, captions, keys, npts=None, measure='cosine', return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    uniq_keys = dict.fromkeys(keys) # preserve original order
    ranks = numpy.zeros(len(uniq_keys))
    top1  = numpy.zeros(len(uniq_keys))

    counter = 0
    for query_idx, query_key in enumerate(uniq_keys):
        # Get query image 
        selected_idx = np.where(keys == query_key)[0]
        selected_image = images[selected_idx][0].reshape(1, images.shape[1])
        #print('selected_idx', selected_idx)
        num_queries = len(selected_idx)

        # compute scores (selected_image v.s. captions)
        similarity_score = np.dot(selected_image, captions.T).flatten()
        similarity_score_inds = numpy.argsort(similarity_score)[::-1]

        # Score
        rank = 1e20
        for i in range(counter, counter + num_queries, 1): 
            tmp = np.where(similarity_score_inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        counter = counter + num_queries
        ranks[query_idx] = rank
        top1[query_idx] = similarity_score_inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)

def test_trees(model_path):
    """ use the trained model to generate parse trees for text """
    # load model and options
    checkpoint = torch.load(model_path, map_location='cpu')
    opt = checkpoint['opt']

    # load vocabulary used by the model
    vocab = pickle.load(open(os.path.join(opt.data_path, 'vocab.pkl'), 'rb'))
    opt.vocab_size = len(vocab)

    # construct model
    model = VGNSL(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])

    print('Loading dataset')
    data_loader = get_eval_loader(
        opt.data_path, 'test', vocab, opt.batch_size, opt.workers, 
        load_img=False, img_dim=opt.img_dim
    )

    cap_embs = None
    logged = False
    trees = list()
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = print
        lengths = torch.Tensor(lengths).long()
        if torch.cuda.is_available():
            lengths = lengths.cuda()

        # compute the embeddings
        model_output = model.forward_emb(images, captions, lengths, volatile=True)
        img_emb, cap_span_features, left_span_features, right_span_features, word_embs, tree_indices, all_probs, \
        span_bounds = model_output[:8]

        candidate_trees = list()
        for j in range(len(ids)):
            candidate_trees.append(generate_tree(captions, tree_indices, j, vocab))
        appended_trees = ['' for _ in range(len(ids))]
        for j in range(len(ids)):
            appended_trees[ids[j] - min(ids)] = clean_tree(candidate_trees[j])
        trees.extend(appended_trees)
        cap_emb = torch.cat([cap_span_features[l-2][i].reshape(1, -1) for i, l in enumerate(lengths)], dim=0)
        del images, captions, img_emb, cap_emb

    ground_truth = [line.strip() for line in open(
        os.path.join(opt.data_path, 'test_ground-truth.txt'))]
    return trees, ground_truth
