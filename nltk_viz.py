from nltk import Tree
from IPython.display import display

import regex
from nltk.draw import TreeView
import os

def viz_tree(bare_tree):
    nt_tree = bare_tree.replace('(', '(NT').replace(' ', '  ')
    #print(nt_tree)
    nt_tree = regex.sub(r' ([^ \(\)]+) ', r' (PT \1) ', nt_tree)



    t = Tree.fromstring('(S (NP this tree) (VP (V is) (AdjP pretty)))')
    TreeView(t)._cframe.print_to_file('output.ps')
    #nltk_tree = Tree.fromstring(nt_tree)
    #nltk_tree.pretty_print()
    #TreeView(nltk_tree)._cframe.print_to_file('tree1.ps')
    #os.system('convert tree1.ps tree1.png')

if __name__ == '__main__':
    bare_tree = '( ( a cat ) meows )'
    viz_tree(bare_tree)

