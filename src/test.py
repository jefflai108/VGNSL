import argparse
import os

from evaluation import test_trees, f1_score
from vocab import Vocabulary

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../data/mscoco',
                        help='path to datasets')
    parser.add_argument('--candidate', type=str, required=True,
                        help='model path to evaluate')
    parser.add_argument('--image_hdf5', help='path to pre-stored image embedding .h5 file')
    parser.add_argument('--data_summary_json', help='karpathy split json file')
    parser.add_argument('--basename', help='MSCOCO split')
    parser.add_argument('--vocab_path', default='../data/mscoco/vocab.pkl',
                        help='path to vocab.pkl')
    parser.add_argument('--visual_tree', '-v', action="store_true", 
                        help='visualize tress')
    parser.add_argument('--visual_samples', type=int, default=10, 
                        help='number of trees to visualize')
    args = parser.parse_args()

    trees, ground_truth, captions = test_trees(args.data_path, args.candidate, args.vocab_path, args.basename, args.visual_tree, args.visual_samples)
    f1, _, _ = f1_score(trees, ground_truth, captions, args.visual_tree)
    #print('Model:', args.candidate)
    #print('F1 score:', f1)
    print(f1)
