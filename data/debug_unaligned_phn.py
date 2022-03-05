

phn_tree_pth = 'data/SpokenCOCO/Freda-formatting/val_phn-level-ground-truth-83k-5k-5k.txt'
phn_caps_pth = 'data/SpokenCOCO/Freda-formatting/val_phn_caps-83k-5k-5k.txt'
with open(phn_tree_pth, 'r') as f: 
    phn_trees = f.readlines()
phn_trees = [x.strip('\n') for x in phn_trees]

with open(phn_caps_pth, 'r') as f: 
    phn_caps = f.readlines()
phn_caps = [x.strip('\n') for x in phn_caps]

def _retrieve_sentence_from_tree(tree): 
    return ' '.join([_x for _x in tree.split() if _x.isalnum()])

for i, phn_tree in enumerate(phn_trees): 
    text_from_phn_tree = _retrieve_sentence_from_tree(phn_tree)
    if phn_tree == 'MISMATCH':
        print('MISMATCH found at %d', i)
        continue
    assert text_from_phn_tree == phn_caps[i], print(text_from_phn_tree, '\n', phn_caps[i])


