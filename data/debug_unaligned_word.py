

word_tree_pth = 'data/SpokenCOCO/Freda-formatting/test_word-level-ground-truth-83k-5k-5k.txt'
word_caps_pth = 'data/SpokenCOCO/Freda-formatting/test_caps-83k-5k-5k.txt'
with open(word_tree_pth, 'r') as f: 
    word_trees = f.readlines()
word_trees = [x.strip('\n') for x in word_trees]

with open(word_caps_pth, 'r') as f: 
    word_caps = f.readlines()
word_caps = [x.strip('\n') for x in word_caps]

def _retrieve_sentence_from_tree(tree): 
    return ' '.join([_x for _x in tree.split() if _x.isalnum()])

for i, word_tree in enumerate(word_trees): 
    text_from_word_tree = _retrieve_sentence_from_tree(word_tree)
    if word_tree == 'MISMATCH':
        print('MISMATCH found at %d', i)
        continue
    assert text_from_word_tree == word_caps[i], print(text_from_word_tree, '\n', word_caps[i])


