import torch 
import torch.nn as nn
import torch.nn.functional as F

from module import AttentivePooling
from utils import l2norm

class SegmentSimilarityMeasure(nn.Module): 
    """
    Compute how similar two segments are. Use MLP instead of dot-product. 
    """
    def __init__(self, hidden_dim): 
        super(SegmentSimilarityMeasure, self).__init__()

        self.sim_measure = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
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
        S = self.sim_measure(concat_batch_segment)

        return S.squeeze(-1)

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

    relative_dissimilarity = torch.ones(relative_similarity.shape) - relative_similarity
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
            torch.maximum(first_order_detector, second_order_detector) - threshold * torch.ones(segment_dissimilarity.shape), \
            torch.zeros_like(segment_dissimilarity) \
        ), first_order_detector \
    )
    return peak_detector 


class DifferentialWordSegmentation(nn.Module): 
    def __init__(self, hidden_dim, peak_detection_threshold, **kwargs):
        super(DifferentialWordSegmentation, self).__init__()
        
        self.adjacent_segment_sim_measure = SegmentSimilarityMeasure(hidden_dim)
        self.peak_detection_threshold = peak_detection_threshold
        self.word_enc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, segment_rep, phn_mask):
        '''
        input:
        segment_rep : size (B, N, H), B: batch size, N: number of phns, H: Hidden dimension
        phn_mask : size (B, N)
        
        return:
        word_segment_rep: size (B, H)
        '''
        batch_size, num_phns, hidden_dim = segment_rep.shape
        
        print(phn_mask)
        # 1. Compute D, phn-segment-wise dissimilarity 
        S = self.adjacent_segment_sim_measure(segment_rep[:, :-1, :], segment_rep[:, 1:, :])
        print(S.shape) # B, N-1
        D = relatvie_segment_dissimilarity(S)
        print(D.shape) # B, N-1
    
        # 2. Run first/second-order peak detector on D 
        P = segment_peak_detector(D, self.peak_detection_threshold)
        P = F.pad(P, (0, 1), value=0) # for matching input_shape. Since there's no next frame for last, Pt should be 0. 
        print(P.shape) # B, N 
        # 2.1 filter out padded phn from P accroding to phn_mask
        P = P + torch.sub(phn_mask, torch.ones_like(phn_mask))
        F.relu(P, inplace=True)
        #print(P)

        # 3. make P binary and differentiable via straight-through
        b_soft = torch.tanh(10 * P)
        b_hard = torch.tanh(100000 * P)
        b = b_soft + (b_hard - b_soft).detach()
        print(b.shape) # B, N 
        #print(b)

        # 4. vectorize b for mean-pooling 
        b = torch.cumsum(b, dim=1) 
        for i in range(batch_size): 
            if b[i][0] == 0: # detect and fix first segments
                b[i] = torch.add(b[i], 1)
        num_word_segments = b[:, -1].int() + 1
        U = -100000 * torch.ones(batch_size, num_phns, max(num_word_segments))
        for i in range(batch_size): 
            num_word_segment = num_word_segments[i]
            U[i, :, :num_word_segment] = torch.arange(1, num_word_segment+1).repeat(num_phns, 1)
        print(U.shape) # B, N, # of segments
        V = U - b.unsqueeze(-1)
        #print(V)
        print(V.shape) # B, N, # of segments 
        W = 1 - torch.tanh(100000 * torch.abs(V))
        W = W / torch.sum(W, dim=1).unsqueeze(1)
        W = torch.nan_to_num(W)

        # these two are good way to debug 
        #print(b[-1])
        #print(W[-1])
        print(W.shape) # B, N, # of segments 

        # 5. word segment representation via W
        W = W.permute(0, 2, 1) # B, # of segments, N 
        word_segment_rep = torch.bmm(W, segment_rep)
        print(word_segment_rep.shape)
        #print(word_segment_rep[-1])
        word_masks = torch.where(word_segment_rep==0, 0, 1)
        #print(word_masks[-1])
        word_segment_rep = self.word_enc(word_segment_rep)
        word_segment_rep = torch.mul(word_segment_rep, word_masks)
        
        return word_segment_rep

if __name__ == '__main__': 
    batch_size = 10
    num_word = 20
    frame_per_segment = 15
    feat_dim = 768
    hidden_dim = 512
    word_segment_threshold = 0.03
    
    x = torch.randn(batch_size * num_word, frame_per_segment, feat_dim)
    audio_mask = torch.ones(batch_size, num_word, frame_per_segment)
    audio_mask[:, -10:, :] = -100000 # half of the phns are masked out 
    audio_mask_reshaped = audio_mask.reshape(-1, frame_per_segment)

    sem_embedding = AttentivePooling(feat_dim, hidden_dim)
    sem_embeddings = sem_embedding(x, audio_mask_reshaped)
    print(sem_embeddings.shape) # torch.Size([480, 512])
    sem_embeddings = sem_embeddings.reshape(batch_size, num_word, hidden_dim)
    print(sem_embeddings.shape) # torch.Size([10, 48, 512])
     
    differential_boundary_module = DifferentialWordSegmentation(hidden_dim, word_segment_threshold)
    differential_boundary_module(sem_embeddings, audio_mask[:, :, 0])

