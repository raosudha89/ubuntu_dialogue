import sys, time
import numpy as np
import cPickle
from collections import OrderedDict, defaultdict
import pdb
from sklearn.decomposition import PCA

TRIGRAM_COUNT_CUTOFF = 30

def get_trigram_vocab(vocab_idx):
    trigram_vocab = defaultdict(int)
    for w in vocab_idx.keys():
        w = "#"+w+"#"
        for i in range(len(w)-2):
            trigram_vocab[w[i:i+3]] += 1
    return trigram_vocab

def get_trigram_vectors(vocab_idx, vocab_size, trigram_vocab_idx, TRIGRAM_UNK_idx):
    trigram_vectors = np.zeros((vocab_size, len(trigram_vocab_idx)+1), dtype=np.int32)
    for w, idx in vocab_idx.iteritems():
        w = "#"+w+"#"
        for j in range(len(w)-2):
            trigram = w[j:j+3]
            try:
                trigram_idx = trigram_vocab_idx[trigram]
            except KeyError:
                trigram_idx = TRIGRAM_UNK_idx
            trigram_vectors[idx][trigram_idx] += 1
    return trigram_vectors

if __name__ == '__main__':
    np.set_printoptions(linewidth=160)
    if len(sys.argv) < 2:
        print "usage: python generate_trigram_embeddings.py vocab_idx.p vocab_size.p trigram_embeddings.p"
        sys.exit(0)
    start_time = time.time()
    print 'loading data...'
    vocab_idx = cPickle.load(open(sys.argv[1], 'rb'))
    vocab_size = cPickle.load(open(sys.argv[2], 'rb'))
    trigram_vocab = get_trigram_vocab(vocab_idx)
    print("Entire Trigram Vocab size %s" % len(trigram_vocab))
    trigram_vocab = OrderedDict(sorted(trigram_vocab.items(), key=lambda t: t[1], reverse=True))
    trigram_vocab = {w:ct for w,ct in trigram_vocab.iteritems() if ct > TRIGRAM_COUNT_CUTOFF}
    trigram_vocab_size = len(trigram_vocab)
    print("Used Trigram Vocab size %s" % trigram_vocab_size)
    idx = 0
    trigram_vocab_idx = {}
    for w, ct in trigram_vocab.iteritems():
        trigram_vocab_idx[w] = idx
        idx += 1
    TRIGRAM_UNK_idx = idx
    trigram_vocab_size += 1 #for trigram UNK token
    trigram_vectors = get_trigram_vectors(vocab_idx, vocab_size, trigram_vocab_idx, TRIGRAM_UNK_idx)
    # pca = PCA(n_components='mle', svd_solver='full')
    pca = PCA(n_components=50)
    pca.fit(trigram_vectors)
    trigram_embeddings = pca.transform(trigram_vectors)
    cPickle.dump(trigram_embeddings, open(sys.argv[3], 'wb'))
    