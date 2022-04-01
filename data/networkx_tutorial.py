import networkx as nx
import numpy as np
from networkx.algorithms.bipartite.matrix import from_biadjacency_matrix
from scipy.sparse import csr_matrix

def generate_sim_mat(gt_word_list, jason_word_feature): 
    # L1 distance (in seconds) between center of each gt and Jason's word segments 
    # gt_word_list example: 
    # [('a', 0.62, 0.81), ('toucan', 0.81, 1.34), ('with', 1.34, 1.52), ('a', 1.52, 1.57), ('brightly', 1.57, 1.98), ('colored', 1.98, 2.34), ('beak', 2.73, 2.83), ('in', 2.83, 2.99), ('a', 2.99, 3.04), ('cage', 3.04, 3.48)]
    # 
    # jason_word_feature example: 
    # tensor([[0.8646, 1.0456],
    #        [1.0858, 1.3472],
    #        [1.4276, 1.4879],
    #        [1.5885, 1.8298],
    #        [1.8499, 1.9504],
    #        [2.0308, 2.2319],
    #        [2.4330, 2.7145],
    #        [2.8955, 2.9759],
    #        [3.0965, 3.3780]], dtype=torch.float64)

    n = len(gt_word_list) 
    m = len(jason_word_feature)
    l1_dist_mat = 1000 * np.ones((n, m))
    
    # retrieve center of each gt and Jason's word segments
    gt_word_list_center = [np.mean([x[1], x[2]]) for x in gt_word_list]
    jason_word_feature_center = np.mean(np.array(jason_word_feature), axis=1)
    
    # compute l1_distance 
    for i in range(n): 
        for j in range(m): 
            l1_dist_mat[i,j] = np.abs(gt_word_list_center[i] - jason_word_feature_center[j])

    return l1_dist_mat

def _permute(edge, sim_mat):
    # Edge not in l,r order. Fix it
    if edge[0] < sim_mat.shape[0]:
        return edge[0], edge[1] - sim_mat.shape[0]
    else:
        return edge[1], edge[0] - sim_mat.shape[0]

def run(gt_word_list, jason_word_feature): 
    # return max weight matching nodes from a bipartite graph. 
    # distance + min-match == -distance + max-match 
    # 
    # reference https://github.com/cisnlp/simalign/blob/05332bf2f6ccde075c3aba94248d6105d9f95a00/simalign/simalign.py#L96-L103

    dist_mat = generate_sim_mat(gt_word_list, jason_word_feature)
    #sim_mat = np.reciprocal(dist_mat) # could have issue with 0 inverse
    sim_mat = -1 * dist_mat

    G = from_biadjacency_matrix(csr_matrix(sim_mat))
    matching = nx.max_weight_matching(G, maxcardinality=True)
    matching = [_permute(x, sim_mat) for x in matching]

    return matching 

if __name__ == '__main__': 
    np.random.seed(0)
    distance = np.random.rand(3, 5)
    sim = np.reciprocal(distance)
    import torch 
    gt_word_list = [('a', 0.35, 0.46), ('black', 0.46, 0.75), ('and', 0.75, 0.88), ('yellow', 0.88, 1.08), ('bird', 1.08, 1.54), ('with', 1.58, 1.73), ('a', 1.73, 1.79), ('huge', 1.79, 2.43), ('colorful', 2.71, 3.13), ('beak', 3.13, 3.47), ('in', 3.47, 3.66), ('a', 3.66, 3.73), ('cage', 3.73, 4.27)]
    jason_word_feature = torch.tensor([[0.5026, 0.8243], [0.8645, 1.0856],[1.2062, 1.3872], [1.5681, 1.7289], [1.8697, 2.1913], [2.3924, 2.4125], [2.7542, 2.8950], [2.9955, 3.1362], [3.1965, 3.4378], [3.5182, 3.6589], [3.7996, 4.1012]], dtype=torch.float64)
    # alignment results: [(6, 5), (12, 10), (7, 4), (11, 7), (4, 2), (9, 8), (8, 6), (3, 1), (5, 3), (1, 0), (10, 9)]

    alignment = run(gt_word_list, jason_word_feature)
    print(alignment)

    """
    Expected output:
    [[0.5488135  0.71518937 0.60276338 0.54488318 0.4236548 ]
     [0.64589411 0.43758721 0.891773   0.96366276 0.38344152]
     [0.79172504 0.52889492 0.56804456 0.92559664 0.07103606]]
    [(1, 2), (2, 3), (0, 1)]
    """
