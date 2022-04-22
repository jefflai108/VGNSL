import random


def left_branching(st):
    words = st.replace('(', '').replace(')', '').split()
    if len(words) == 1:
        return (f'( {words[0]} )')
    else:
        current_st = f'( {words[0]} {words[1]} )'
        for item in words[2:]:
            current_st = f'( {current_st} {item} )'
        return current_st


def right_branching(st):
    words = st.replace('(', '').replace(')', '').split()
    if len(words) == 1:
        return (f'( {words[0]} )')
    else:
        current_st = f'( {words[-2]} {words[-1]} )'
        for item in words[2:]:
            current_st = f'( {item} {current_st} )'
        return current_st


def random_branching(st, seed=0):
    random.seed(seed)
    words = st.replace('(', '').replace(')', '').split()
    while len(words) > 1:
        position = random.randint(0, len(words) - 2)
        item = f'( {words[position]} {words[position+1]} )'
        words = words[:position] + [item] + words[position+2:]
    return words[0]


def eval_baselines(fname):
    from metrics import corpus_f1, sentence_f1
    gold_trees = [x.strip() for x in open(fname)]
    gold_trees[:] = [sentence for sentence in gold_trees if sentence != 'MISMATCH'] # remove mismatch
    print(f'File Name: {fname}')
    # Left
    left_trees = [left_branching(x) for x in gold_trees]
    print(f'Left branching')
    print(f'\tCorpus F1: {corpus_f1(left_trees, gold_trees)*100:.3f}')
    print(f'\tSentence F1: {sentence_f1(left_trees, gold_trees)*100:.3f}')
    # Right
    right_trees = [right_branching(x) for x in gold_trees]
    print(f'Right branching')
    print(f'\tCorpus F1: {corpus_f1(right_trees, gold_trees)*100:.3f}')
    print(f'\tSentence F1: {sentence_f1(right_trees, gold_trees)*100:.3f}')
    # Random 
    c_f1 = 0
    s_f1 = 0
    for seed in range(5):
        random_trees = [random_branching(x, seed) for x in gold_trees]
        c_f1 += corpus_f1(random_trees, gold_trees)
        s_f1 += sentence_f1(random_trees, gold_trees)
    c_f1 /= 5
    s_f1 /= 5
    print(f'Random branching')
    print(f'\tCorpus F1: {c_f1*100:.3f}')
    print(f'\tSentence F1: {s_f1*100:.3f}')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_tree_pth", type=str)
    args = parser.parse_args()

    eval_baselines(args.gt_tree_pth)

