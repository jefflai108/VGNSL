# Create a vocabulary wrapper
# source: https://github.com/ExplorerFreda/LatentTree/blob/2f506a6f60a04bc13df3b75612c6e163e58a3745/VSE/vocab.py
import nltk
import pickle
from collections import Counter
import json
import os

class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(transcript_list, threshold):
    """Build a simple vocabulary wrapper."""
    counter = Counter()
    for i, transcript in enumerate(transcript_list):
        tokens = nltk.tokenize.word_tokenize(transcript.lower())
        counter.update(tokens)

        if i % 1000 == 0:
            print("[%d/%d] tokenized the captions." % (i, len(transcript_list)))

    # Discard if the occurrence of the word is less than min_word_cnt.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab
