import argparse
import os

from evaluation import test_trees, f1_score
from vocab import Vocabulary

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../data/mscoco',
                        help='path to datasets')
    parser.add_argument('--export_tree', action="store_true",
                        help='export inferred tree to file')
    parser.add_argument('--export_tree_path',
                        help='path to store the inferred trees')
    parser.add_argument('--candidate', type=str, required=True,
                        help='model path to evaluate')
    parser.add_argument('--image_hdf5', help='path to pre-stored image embedding .h5 file')
    parser.add_argument('--data_summary_json', help='karpathy split json file')
    parser.add_argument('--basename', help='MSCOCO split')
    parser.add_argument('--data_split', type=str, default='test',
                        help='targeted data_split', choices = ['train', 'val', 'test'])
    parser.add_argument('--vocab_path', default='../data/mscoco/vocab.pkl',
                        help='path to vocab.pkl')
    parser.add_argument('--visual_tree', '-v', action="store_true", 
                        help='visualize tress')
    parser.add_argument('--visual_samples', type=int, default=10, 
                        help='number of trees to visualize')
    parser.add_argument('--constituent_recall', action="store_true", 
                        help='run constituency recall on predicted trees.')
    args = parser.parse_args()

    trees, ground_truth, captions = test_trees(args.data_path, args.candidate, args.vocab_path, args.basename, args.data_split, 
                                               args.visual_tree, args.visual_samples, args.export_tree, args.export_tree_path)
    f1, _, _ = f1_score(trees, ground_truth, captions, args.visual_tree, args.constituent_recall)
    #print('Model:', args.candidate)
    #print('F1 score:', f1)
    if args.visual_tree:
        print('visual samples f1 score is %f' % f1)
    else: print(f1)
