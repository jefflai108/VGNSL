import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

from utils import l2norm

class SegmentSimilarityMeasure(nn.Module): 
    """
    Compute how similar two segments are. Use MLP instead of dot-product. 
    """
    def __init__(self, hidden_dim): 
        super(SegmentSimilarityMeasure, self).__init__()

        self.sim_measure = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, batch_segment1, batch_segment2): 
        """
        input: 
        batch_segment1: (B, N-1, H), B: batch size, N: number of phns, H: Hidden dimension
        batch_segment2: (B, N-1, H)

        output: 
        batch_sim: (B, N-1)
        """
        concat_batch_segment = torch.cat(
            (l2norm(batch_segment1), l2norm(batch_segment2)), 
            dim=2
        )
        S = self.sim_measure(concat_batch_segment).squeeze(-1)
        S = F.sigmoid(S)

        return S

def relatvie_segment_dissimilarity(segment_similarity): 
    """
    input: 
    S: size (B, N-1), B: batch size, N: number of phns

    output:
    D: size (B, N-1), where D = 1 - (S - min(S)) / (max(S) - min (S)) 
    """
    min_segment_similarity, _  = torch.min(segment_similarity, dim=1)
    max_segment_similarity, _  = torch.max(segment_similarity, dim=1)

    relative_similarity = torch.div((segment_similarity - min_segment_similarity[:, None]), \
                                    (max_segment_similarity[:, None] - min_segment_similarity[:, None]))

    relative_dissimilarity = torch.ones_like(relative_similarity) - relative_similarity
    return relative_dissimilarity

def segment_peak_detector(segment_dissimilarity, threshold): 
    """
    peak detection over words/phns based on dissimilarity measure, including first-order & second-order.

    input: 
    D: size (B, N-1), B: batch size, N: number of phns

    output: 
    P: size (B, N-1)
    """
    # peak detectors are based on element-wise min/max operator

    # first-oreder p_1
    first_order_detector = torch.zeros_like(segment_dissimilarity)
    first_order_detector[:, 0] = torch.maximum((segment_dissimilarity[:, 0] - segment_dissimilarity[:, 1]), torch.zeros_like(segment_dissimilarity[:, 0]))
    first_order_detector[:, 1:-1] = torch.minimum( \
        torch.maximum((segment_dissimilarity[:, 1:-1] - segment_dissimilarity[:, :-2]), torch.zeros_like(segment_dissimilarity[:, 1:-1])), \
        torch.maximum((segment_dissimilarity[:, 1:-1] - segment_dissimilarity[:, 2:]), torch.zeros_like(segment_dissimilarity[:, 1:-1])) \
    ) # torch.Size([10, 45])
    first_order_detector[:, -1] = torch.maximum((segment_dissimilarity[:, -1] - segment_dissimilarity[:, -2]), torch.zeros_like(segment_dissimilarity[:, -1]))

    # second-order p_2
    second_order_detector = torch.zeros_like(segment_dissimilarity)
    second_order_detector[:, 0:2] = torch.maximum((segment_dissimilarity[:, 0:2] - segment_dissimilarity[:, 2:4]), torch.zeros_like(segment_dissimilarity[:, 0:2]))
    second_order_detector[:, 2:-2] = torch.minimum( \
        torch.maximum((segment_dissimilarity[:, 2:-2] - segment_dissimilarity[:, :-4]), torch.zeros_like(segment_dissimilarity[:, 2:-2])), \
        torch.maximum((segment_dissimilarity[:, 2:-2] - segment_dissimilarity[:, 4:]), torch.zeros_like(segment_dissimilarity[:, 2:-2])) \
    ) # torch.Size([10, 43])
    first_order_detector[:, -2:] = torch.maximum((segment_dissimilarity[:, -2:] - segment_dissimilarity[:, -4:-2]), torch.zeros_like(segment_dissimilarity[:, -2:]))

    # p based on p_1, p_2, threshold
    peak_detector = torch.minimum( \
        torch.maximum( \
            torch.maximum(first_order_detector, second_order_detector) - threshold * torch.ones_like(segment_dissimilarity), \
            torch.zeros_like(segment_dissimilarity) \
        ), first_order_detector \
    )
    return peak_detector 

def weighted_random_peak_sampling(P, phn_mask, gt_word_lens, proxy_large_num=10):
    """
    weighted_random sample top_k selection on peak to match # of words specified in phn_mask

    input: 
    P: size (B, N), B: batch size, N: number of phns
    phn_mask : size (B, N)
    gt_word_lens: size (B,)

    output: 
    P: size (B, N)
    """
    # iterate through P as each Pi has different # of gt words 
    # ensure top-k word segments are selected from *the valid phn ranges*
    for i in range(len(P)): 
        num_of_words = gt_word_lens[i]
        num_of_phns = torch.sum(torch.where(phn_mask[i] != -100000, 1, 0)).item()
        # select indices via weighted sampling based on P[i, :num_of_phns]
        weights = P[i, :num_of_phns]
        topk_indices = torch.multinomial(weights, num_of_words, replacement=False)
        P[i, topk_indices] = proxy_large_num
   
    P = torch.where(P != proxy_large_num, torch.tensor(0.0, device=P.device), torch.tensor(1.0, device=P.device)) # rest of P set to 0. This will avoid over-segmentation.
    # good way to debug
    #print(P)
    #print(gt_word_lens)
    return P
   
class DifferentialWordSegmentation(nn.Module): 
    def __init__(self, hidden_dim, peak_detection_threshold):
        super(DifferentialWordSegmentation, self).__init__()
        
        self.adjacent_segment_sim_measure = SegmentSimilarityMeasure(hidden_dim)
        self.peak_detection_threshold = peak_detection_threshold
        self.word_transform = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, segment_rep, phn_mask, gt_word_lens):
        '''
        input:
        segment_rep : size (B, N, H), B: batch size, N: number of phns, H: Hidden dimension
        phn_mask : size (B, N)
        gt_word_lens: size (B,)

        return:
        word_segment_rep: size (B, M, H), M: max number of words in the minibatch
        '''
        batch_size, num_phns, hidden_dim = segment_rep.shape
        
        #print(phn_mask)
        # 1. Compute D, phn-segment-wise dissimilarity 
        S = self.adjacent_segment_sim_measure(segment_rep[:, :-1, :], segment_rep[:, 1:, :])
        #print(S.shape) # B, N-1
        D = relatvie_segment_dissimilarity(S)
        #print(D.shape) # B, N-1
    
        # 2. Run first/second-order peak detector on D 
        P = segment_peak_detector(D, self.peak_detection_threshold)
        P = F.pad(P, (0, 1), value=0) # for matching input_shape. Since there's no next frame for last, Pt should be 0.
        #print(P.shape) # B, N
        
        # 2.1 filter out padded phn from P accroding to phn_mask
        #P = P + torch.sub(phn_mask, torch.ones_like(phn_mask)) # phn_mask consists of 1 and 1e-5
        #print(P)
        P = P + phn_mask # phn_mask consists of 0 and 1e-5
        with torch.no_grad():
            P = F.relu(P)
        #print(gt_word_lens)

        # 2.2 top-k sampling for enforcing # of word segment
        P = weighted_random_peak_sampling(P, phn_mask, gt_word_lens)
        #print(P)
        
        # 3. make P binary and differentiable via straight-through
        b_soft = torch.tanh(P)
        b_hard = torch.tanh(100000 * P)
        b = b_soft + (b_hard - b_soft).detach()
        #print(b.shape) # B, N 

        # these two are good way to debug 
        #print(b)
        #print(gt_word_lens)

        # 4. vectorize b for mean-pooling 
        b = torch.cumsum(b, dim=1) 
        #print(b)
        assert torch.all(b[:, -1] == gt_word_lens), print('num of word segments should equal the number of words')
        num_word_segments = b[:, -1].int()
        U = -100000 * torch.ones(batch_size, num_phns, max(num_word_segments), device=b.device)
        for i in range(batch_size): 
            num_word_segment = num_word_segments[i]
            U[i, :, :num_word_segment] = torch.arange(1, num_word_segment+1).repeat(num_phns, 1)
        #print(U.shape) # B, N, M 
        V = U - b.unsqueeze(-1)
        #print(V)
        #print(V.shape) # B, N, M
        W = 1 - torch.tanh(100000 * torch.abs(V))
        W = W / torch.sum(W, dim=1).unsqueeze(1)
        W = torch.nan_to_num(W)
        # these two are good way to debug 
        #print(b[-2])
        #print(W[-2])
        #print(gt_word_lens)
        #print(W.shape) # B, N, M

        # 5. word segment representation via W
        W = W.permute(0, 2, 1) # B, M, N 
        word_segment_rep = torch.bmm(W, segment_rep)
        #print(word_segment_rep.shape) # B, M, H
        word_segment_rep = self.word_transform(word_segment_rep)
        #print(word_segment_rep[-1])
        word_masks = torch.where(word_segment_rep==0, 0, 1)
        #print(word_masks[-1])
        #print(word_segment_rep.shape) # B, M, H
        #print(word_segment_rep[-1, 5:])
        return word_segment_rep, (b.data.cpu().numpy(), gt_word_lens.data.cpu().numpy())

if __name__ == '__main__': 
    batch_size = 10
    num_word = 20
    frame_per_segment = 15
    feat_dim = 768
    hidden_dim = 512
    word_segment_threshold = 0.00 # for enforcing # of words 
    gt_word_lens = torch.tensor([2, 3, 5, 2, 6, 3, 4, 9, 7, 4])
    
    x = torch.randn(batch_size * num_word, frame_per_segment, feat_dim)
    audio_mask = torch.ones(batch_size, num_word, frame_per_segment)
    audio_mask[:-1, -10:, :] = -100000 # half of the phns are masked out 
    audio_mask_reshaped = audio_mask.reshape(-1, frame_per_segment)

    #sem_embedding = AttentivePooling(feat_dim, hidden_dim)
    sem_embedding = MeanPooledPhoneSegment()
    sem_embeddings = sem_embedding(x, audio_mask_reshaped)
    print(sem_embeddings.shape) # torch.Size([200, 768])
    sem_embeddings = sem_embeddings.reshape(batch_size, num_word, feat_dim)
    print(sem_embeddings.shape) # torch.Size([10, 20, 768])
    
    differential_boundary_module = DifferentialWordSegmentation(hidden_dim, word_segment_threshold)
    sem_embeddings, (b, gt_word_lens) = differential_boundary_module(sem_embeddings, audio_mask[:, :, 0], gt_word_lens)
    print(sem_embeddings.shape)
    print(b)
    print(gt_word_lens)

