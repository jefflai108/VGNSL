from nltk import Tree


class BareTree(Tree):
    @classmethod
    def fromstring(cls, st):
        augmented_st = st.replace('(', '( NT')
        return super().fromstring(augmented_st)

    @staticmethod
    def spans(node, left=0):
        if isinstance(node, str):
            return
        yield (left, left + len(node.leaves()))
        child_left = left
        for child in node:
            for x in BareTree.spans(child, child_left):
                yield x
            n_child_leaves = 1 if isinstance(child, str) else len(child.leaves())
            child_left += n_child_leaves


def corpus_f1(pred_trees, gold_trees):
    cnt_spans_prec = 0
    cnt_spans_rec = 0
    cnt_spans_matched = 0
    for pred, gold in zip(pred_trees, gold_trees):
        if len(gold.split()) == 1: # single word only (very rare)
            continue 
        pred_spans = set(BareTree.spans(BareTree.fromstring(pred)))
        gold_spans = set(BareTree.spans(BareTree.fromstring(gold)))
        matched_spans = pred_spans.intersection(gold_spans)
        n_matched = len(matched_spans)
        n_prec = len(pred_spans)
        n_rec = len(gold_spans)
        cnt_spans_matched += n_matched
        cnt_spans_prec += n_prec
        cnt_spans_rec += n_rec
    prec = cnt_spans_matched / cnt_spans_prec
    rec = cnt_spans_matched / cnt_spans_rec
    f1_score = 2 * prec * rec / (rec + prec)
    return f1_score


def sentence_f1(pred_trees, gold_trees):
    accu_f1_score = 0
    cnt_instances = 0
    for pred, gold in zip(pred_trees, gold_trees):
        if len(gold.split()) == 1: # single word only (very rare)
            continue 
        pred_spans = set(BareTree.spans(BareTree.fromstring(pred)))
        gold_spans = set(BareTree.spans(BareTree.fromstring(gold)))
        matched_spans = pred_spans.intersection(gold_spans)
        n_matched = len(matched_spans)
        n_prec = len(pred_spans)
        n_rec = len(gold_spans)
        prec = n_matched / n_prec
        rec = n_matched / n_rec
        f1_score = 2 * prec * rec / (rec + prec)
        accu_f1_score += f1_score 
        cnt_instances += 1
    return accu_f1_score / cnt_instances
