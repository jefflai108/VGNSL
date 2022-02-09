from nltk import Tree
import regex


def viz_tree(bare_tree):
    nt_tree = bare_tree.replace('(', '(NT').replace(' ', '  ')
    print(nt_tree)
    nt_tree = regex.sub(r' ([^ \(\)]+) ', r' (PT \1) ', nt_tree)
    nltk_tree = Tree.fromstring(nt_tree)
    nltk_tree.pretty_print()


if __name__ == '__main__':
    bare_tree = '( ( a cat ) meows )'
    viz_tree(bare_tree)

