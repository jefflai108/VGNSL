from nltk import Tree
import numpy as np

class ExNLTKTree(Tree):
    @staticmethod
    def spans(node, left=0):
        if isinstance(node, str):
            return
        if len(node.leaves()) > 1:
            yield (left, left + len(node.leaves()))
        child_left = left
        for child in node:
            for x in BareTree.spans(child, child_left):
                yield x
            n_child_leaves = 1 if isinstance(child, str) else len(child.leaves())
            child_left += n_child_leaves

    def spans_set(self):
        return list(self.spans(self))


class BareTree(ExNLTKTree):
    @classmethod
    def fromstring(cls, st):
        augmented_st = st.replace('(', '( NT ')
        return super().fromstring(augmented_st)


def corpus_f1(pred_trees, gold_trees, aligns, is_baretree=False):
    """
    Compute corpus-level F1 score.
    """
    
    assert len(pred_trees) == len(gold_trees)
    assert len(pred_trees) == len(aligns)

    matched = 0
    gold_cnt = 0
    pred_cnt = 0

    for pred_tree, gold_tree, align in zip(pred_trees, gold_trees, aligns):
        if '(' not in pred_tree or '(' not in gold_tree: # for (rare) cases where there is just one word.
            continue 
        pred_tree = ExNLTKTree.fromstring(pred_tree) if not is_baretree else BareTree.fromstring(pred_tree)
        gold_tree = ExNLTKTree.fromstring(gold_tree) if not is_baretree else BareTree.fromstring(gold_tree)
        pred2gold_align = {k: v for k, v in align}
        pred_spans = pred_tree.spans_set()
        gold_spans = gold_tree.spans_set()
        mapped_pred_spans = set([(pred2gold_align.get(l, -1), pred2gold_align.get(r-1, -1)+1) for l, r in pred_spans])
        matched_spans = mapped_pred_spans.intersection(gold_spans)
        this_matched = len(matched_spans)
        this_gold = len(gold_spans)
        this_pred = len(pred_spans)
        matched += this_matched
        gold_cnt += this_gold
        pred_cnt += this_pred

    if matched == 0:
        return 0.0
    else:
        recall = 100.0 * matched / gold_cnt 
        precision = 100.0 * matched / pred_cnt
        f1_score = 2 * recall * precision / (recall + precision)
        return f1_score


def sentence_f1(pred_trees, gold_trees, aligns, is_baretree=False):
    """
    Compute sentence-level F1 score.
    """
    assert len(pred_trees) == len(gold_trees)
    assert len(pred_trees) == len(aligns)

    sentence_f1s = list()

    for pred_tree, gold_tree, align in zip(pred_trees, gold_trees, aligns):
        sentence_f1s.append(corpus_f1([pred_tree], [gold_tree], [align], is_baretree))

    return np.mean(sentence_f1s)


if __name__ == '__main__':
    pred_trees = ['((1 x) 2)']
    gold_trees = ['(1 2)']
    aligns = [[(0, 0), (2, 1)]]  # positions (pred, gold)
    print(corpus_f1(pred_trees, gold_trees, aligns, is_baretree=True))
    
    pred_trees = ['(NT (NT (PT 1) (PT x)) (PT 2))']
    gold_trees = ['(NT (PT 1) (PT 2))']
    print(corpus_f1(pred_trees, gold_trees, aligns, is_baretree=False))

    pred_trees = ['((1 x) 3)']
    gold_trees = ['(1 (2 3))']
    aligns = [[(0, 0), (2, 2)]]
    print(corpus_f1(pred_trees, gold_trees, aligns, is_baretree=True))

    # our case 
    pred_trees = ['(((s s) s) s)']
    pred_trees = ['(((s s) (s s s) s) s)']
    gold_trees = ['(1 (2 3))']
    aligns = [[(0, 0), (2, 2)]]
    print(corpus_f1(pred_trees, gold_trees, aligns, is_baretree=True))

    # more test    # our case 
    gold_trees = ['( ( a young boy ) barefoot ( holding ( ( an umbrella ) ( touching ( ( the horn ) ( of ( a cow ) ) ) ) ) ) )', '( ( ( a young boy ) ( with ( an umbrella ) ) ) ( who ( is ( touching ( ( the horn ) ( of ( a cow ) ) ) ) ) ) )', '( ( a young man ) ( holding ( an umbrella ) ( next ( to ( ( a herd ) ( of cattle ) ) ) ) ))', '( ( a child ) ( ( holding ( a flowered umbrella ) ) and ( petting ( a yak ) ) ) )', '( ( a boy ) ( holding ( an umbrella ) ( while ( standing ( next ( to livestock ) ) ) ) ) )']
    pred_trees = ['( ( shit ( ( ( shit ( shit ( ( shit shit ) shit ) ) ) shit ) ) ( shit ( shit ( shit shit ) ) ) ) shit )', '( ( ( ( shit shit ) ( shit ( shit shit ) ) ) ( shit ( shit shit ) ) ) shit )', '( shit ( shit ( ( shit shit ) ( ( shit shit ) shit ) ) ) )', '( ( ( ( shit shit ) ( ( shit shit ) shit ) ) ( shit shit ) ) shit )', '( ( ( shit ( shit ( shit ( ( shit shit ) shit ) ) ) ) shit ) ( shit shit ) )']
    aligns = [[(2, 1), (0, 3), (7, 7), (9, 9), (10, 10), (3, 2), (1, 0), (12, 11), (6, 6), (4, 4), (8, 8), (5, 5)], [(6, 4), (1, 0), (5, 3), (3, 2), (10, 6), (8, 5), (13, 8), (11, 7), (2, 1)], [(9, 4), (1, 0), (5, 2), (10, 5), (7, 3), (2, 1), (11, 6)], [(6, 4), (1, 0), (8, 6), (4, 2), (5, 3), (7, 5), (9, 7), (2, 1)], [(0, 0), (1, 1), (6, 4), (7, 5), (8, 6), (5, 3), (9, 8), (3, 2), (4, 7)]]
    aligns_reverse = []
    for align in aligns: 
        aligns_reverse.append([(t[1], t[0]) for t in align])

    print(corpus_f1(gold_trees, pred_trees, aligns, is_baretree=True)) # correct
    print(corpus_f1(pred_trees, gold_trees, aligns, is_baretree=True)) # wrong 
    print(corpus_f1(pred_trees, gold_trees, aligns_reverse, is_baretree=True)) # correct
