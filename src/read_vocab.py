import pickle 
from vocab import Vocabulary

vocab = pickle.load(open('data/SpokenCOCO/SpokenCOCO_vocab.pkl', 'rb'))
print(len(vocab))
vocab = pickle.load(open('data/mscoco/vocab.pkl', 'rb'))
print(len(vocab))
exit()
for i in range(10):
    print(vocab.idx2word[i])

vocab_size = len(vocab)
