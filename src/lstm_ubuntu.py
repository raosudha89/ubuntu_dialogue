import sys
import theano, lasagne, cPickle, time                                                   
import numpy as np
import theano.tensor as T     
from collections import OrderedDict, Counter, defaultdict
import math, random
import pdb
from sklearn.decomposition import PCA

N = 10
DEPTH = 1
TRIGRAM_COUNT_CUTOFF = 30

def iterate_minibatches(contexts, context_masks, responses, response_masks, batch_size, shuffle=False):
    if shuffle:
        indices = np.arange(len(contexts))
        np.random.shuffle(indices)
    for start_idx in range(0, len(contexts) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield contexts[excerpt], context_masks[excerpt], responses[excerpt], response_masks[excerpt]

def iterate_minibatches_train(contexts, context_masks, responses, response_masks, labels, batch_size, shuffle=False):
    if shuffle:
        indices = np.arange(len(contexts))
        np.random.shuffle(indices)
    for start_idx in range(0, len(contexts) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield contexts[excerpt], context_masks[excerpt], responses[excerpt], response_masks[excerpt], labels[excerpt]

def shuffle(r, rm, l):
    shuffled_r = np.zeros(r.shape, dtype=np.int32)
    shuffled_rm = np.zeros(rm.shape, dtype=np.float32)
    shuffled_l = np.zeros(l.shape, dtype=np.int32)

    for i in range(len(r)):
        indexes = range(len(r[i]))
        random.shuffle(indexes)
        for j, index in enumerate(indexes):
            shuffled_r[i][j] = r[i][index]
            shuffled_rm[i][j] = rm[i][index]
            shuffled_l[i][j] = l[i][index]

    return shuffled_r, shuffled_rm, shuffled_l

def validate_train(val_fn, fold_name, epoch, fold, batch_size):
    start = time.time()
    num_batches = 0.
    cost = 0.
    acc = 0.
    contexts, context_masks, responses, response_masks, labels = fold
    responses = responses[:,0,:]
    response_masks = response_masks[:,0,:]
    for c, cm, r, rm, l in iterate_minibatches_train(contexts, context_masks, responses, response_masks, labels,\
                                                batch_size, shuffle=True):
        loss, probs = val_fn(c, cm, r, rm, l)
        corr = 0
        for i in range(len(probs)):
            if probs[i] >= 0.5 and l[i] == 1 or probs[i] < 0.5 and l[i] == 0:
                corr += 1
        acc += corr*1.0/len(probs)    
        cost += loss*1.0/len(probs)
        num_batches += 1
    lstring = '%s: epoch:%d, cost:%f, acc:%f, time:%d' % \
                (fold_name, epoch, cost / num_batches, acc / num_batches, time.time()-start)
    print lstring

def get_rank(preds, labels):
    preds = np.array(preds)
    correct = np.where(labels==1)[0][0]
    sort_index_preds = np.argsort(preds)
    desc_sort_index_preds = sort_index_preds[::-1] #since ascending sort and we want descending
    rank = np.where(desc_sort_index_preds==correct)[0][0]
    return rank+1

def validate(val_fn, fold_name, epoch, fold, batch_size):
    start = time.time()
    num_batches = 0.
    cost = 0.
    acc = 0.
    recall = [0]*N
    contexts, context_masks, responses_list, response_masks_list = fold
    responses_list = responses_list[:,:N,:]
    response_masks_list = response_masks_list[:,:N,:]
    for c, cm, r, rm in iterate_minibatches(contexts, context_masks, responses_list, response_masks_list, \
                                                batch_size, shuffle=True):
        l = np.zeros((batch_size, N), dtype=np.int32)
        l[:,0] = 1
        r, rm, l = shuffle(r, rm, l)
        probs = np.zeros((batch_size, N))
        for j in range(N):
            out = val_fn(c, cm, r[:,j,:], rm[:,j,:], l[:,j])
            loss = out[0]
            # probs[:,j] = out[1][:,0]
            probs[:,j] = out[1]
            cost += loss*1.0/len(probs)
        corr = 0
        for i in range(batch_size):
            rank = get_rank(probs[i], l[i])
            if rank == 1:
                corr += 1
            # if np.argmax(probs[i]) == np.argmax(l[i]):
            #     corr += 1
            for index in range(N):
                if rank <= index+1:
                    recall[index] += 1
        acc += corr*1.0/batch_size
        num_batches += 1
    recall = [round(curr_r*1.0/(batch_size*num_batches), 3) for curr_r in recall]
    lstring = '%s: epoch:%d, cost:%f, acc:%f, time:%d' % \
                (fold_name, epoch, cost / num_batches, acc / num_batches, time.time()-start)
    print lstring
    print recall

def build_lstm(word_embeddings, len_voc, d_word, d_hidden, max_len, batch_size, lr, rho, forget_gate_bias, freeze=False):

    # input theano vars
    contexts = T.imatrix(name='context')
    context_masks = T.matrix(name='context_mask')
    responses = T.imatrix(name='responses')
    response_masks = T.matrix(name='response_masks')
    labels = T.ivector(name='label')   
 
    # define network
    l_context = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=contexts)
    l_context_mask = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=context_masks)
    l_context_emb = lasagne.layers.EmbeddingLayer(l_context, len_voc, d_word, W=word_embeddings)
    # l_context_emb = lasagne.layers.EmbeddingLayer(l_context, len_voc, d_word, W=lasagne.init.GlorotNormal())
    
    l_context_lstm = lasagne.layers.LSTMLayer(l_context_emb, d_hidden, \
                                            grad_clipping=10, \
                                            forgetgate=lasagne.layers.Gate(b=lasagne.init.Constant(forget_gate_bias)), \
                                            mask_input=l_context_mask, \
                                            learn_init=True, \
                                            peepholes=True,\
                                            )

    l_response = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=responses)
    l_response_mask = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=response_masks)
    l_response_emb = lasagne.layers.EmbeddingLayer(l_response, len_voc, d_word, W=l_context_emb.W)
    
    l_response_lstm = lasagne.layers.LSTMLayer(l_response_emb, d_hidden, \
                                            grad_clipping=10, \
                                            # forgetgate=lasagne.layers.Gate(b=lasagne.init.Constant(forget_gate_bias)), \
                                            mask_input=l_response_mask, \
                                            learn_init=True, \
                                            peepholes=True,\
                                            ingate=lasagne.layers.Gate(W_in=l_context_lstm.W_in_to_ingate,\
                                                                W_hid=l_context_lstm.W_hid_to_ingate,\
                                                                b=l_context_lstm.b_ingate,\
                                                                nonlinearity=l_context_lstm.nonlinearity_ingate),\
                                            outgate=lasagne.layers.Gate(W_in=l_context_lstm.W_in_to_outgate,\
                                                                W_hid=l_context_lstm.W_hid_to_outgate,\
                                                                b=l_context_lstm.b_outgate,\
                                                                nonlinearity=l_context_lstm.nonlinearity_outgate),\
                                            forgetgate=lasagne.layers.Gate(W_in=l_context_lstm.W_in_to_forgetgate,\
                                                                W_hid=l_context_lstm.W_hid_to_forgetgate,\
                                                                b=l_context_lstm.b_forgetgate,\
                                                                nonlinearity=l_context_lstm.nonlinearity_forgetgate),\
                                            cell=lasagne.layers.Gate(W_in=l_context_lstm.W_in_to_cell,\
                                                                W_hid=l_context_lstm.W_hid_to_cell,\
                                                                b=l_context_lstm.b_cell,\
                                                                nonlinearity=l_context_lstm.nonlinearity_cell),\
                                            )    

    # add dropout
    l_context_lstm = lasagne.layers.DropoutLayer(l_context_lstm, p=0.4)
    l_response_lstm = lasagne.layers.DropoutLayer(l_response_lstm, p=0.4)

    # now get aggregate embeddings
    context_out = lasagne.layers.get_output(l_context_lstm)
    context_out = T.mean(context_out * context_masks[:,:,None], axis=1)
    
    responses_out = lasagne.layers.get_output(l_response_lstm)
    responses_out = T.mean(responses_out * response_masks[:,:,None], axis=1)
    
    # objective computation
    M = theano.shared(np.eye(d_hidden, dtype=np.float32))

    probs = T.sum(T.dot(context_out, M)*responses_out, axis=1)
    probs = lasagne.nonlinearities.sigmoid(probs)       
    
    loss = T.sum(lasagne.objectives.binary_crossentropy(probs, labels))

    all_context_params = lasagne.layers.get_all_params(l_context_lstm, trainable=True)

    all_params =  all_context_params + [M]
    
    loss += rho * sum(T.sum(l ** 2) for l in all_params)
    
    updates = lasagne.updates.adam(loss, all_params, learning_rate=lr)    
    train_fn = theano.function([contexts, context_masks, responses, response_masks, labels], \
                               [loss, probs], updates=updates)
    val_fn = theano.function([contexts, context_masks, responses, response_masks, labels], \
                               [loss, probs])
    return train_fn, val_fn

def uniform_sample(a, b, k=0):
    if k == 0:
        return random.uniform(a, b)
    ret = np.zeros((k,))
    for x in xrange(k):
        ret[x] = random.uniform(a, b)
    return ret

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
    for i, v in enumerate(trigram_vectors):
        if not np.any(v):
            trigram_vectors[i][TRIGRAM_UNK_idx] = 1 #for UNK and POS tag indices
    return trigram_vectors

def get_trigram_embeddings(vocab_idx, vocab_size):
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
    return trigram_embeddings

def get_word_embeddings(vocab_idx, vocab_size, glove_we, glove_vocab, d_word):
    word_embeddings = [None]*vocab_size
    for w, idx in vocab_idx.iteritems():
        try:
            word_embeddings[idx] = glove_we[glove_vocab[w]]
        except KeyError: #UNK i.e. word not in Glove vocab
            word_embeddings[idx] = uniform_sample(-0.25,0.25,d_word)
    for idx, we in enumerate(word_embeddings):
        if we == None: #POS tags and UNK
            word_embeddings[idx] = uniform_sample(-0.25,0.25,d_word)
    return word_embeddings

if __name__ == '__main__':
    np.set_printoptions(linewidth=160)
    if len(sys.argv) < 4:
        print "usage: python lstm_ubuntu.py train.p dev.p vocab_idx.p vocab_size.p batch_size glove_we glove_vocab"
        sys.exit(0)
    start_time = time.time()
    print 'loading data...'
    vocab_idx = cPickle.load(open(sys.argv[4], 'rb'))
    vocab_size = cPickle.load(open(sys.argv[5], 'rb'))
    batch_size = int(sys.argv[6])
    glove_we = cPickle.load(open(sys.argv[7], 'rb'))
    glove_vocab = cPickle.load(open(sys.argv[8], 'rb'))
    d_word = 300
    d_hidden = 300
    freeze = False
    lr = 0.001
    n_epochs = 5
    rho = 1e-5
    forget_gate_bias = 2.0

    word_embeddings = get_word_embeddings(vocab_idx, vocab_size, glove_we, glove_vocab, d_word)
    word_embeddings = np.asarray(word_embeddings, dtype=np.float32)
    
    trigram_embeddings = get_trigram_embeddings(vocab_idx, vocab_size)
    trigram_embeddings = np.asarray(trigram_embeddings, dtype=np.float32)
    
    print word_embeddings.shape
    print trigram_embeddings.shape
    word_embeddings = np.concatenate((word_embeddings, trigram_embeddings), axis=1)
    d_word += 50
    d_hidden += 50
    
    #word_embeddings = None
    print 'vocab_size', vocab_size
    train = cPickle.load(open(sys.argv[1], 'rb'))
    dev = cPickle.load(open(sys.argv[2], 'rb'))
    test = cPickle.load(open(sys.argv[3], 'rb'))
    max_len = len(train[0][1])
    print 'done loading'
    print 'time taken ', time.time() - start_time

    start_time = time.time()
    print 'compiling graph...'
    train_fn, val_fn = build_lstm(word_embeddings, vocab_size, d_word, d_hidden, max_len, batch_size, lr, rho, forget_gate_bias, freeze=freeze)
    print 'done compiling'
    print 'time taken ', time.time() - start_time

    # train network
    for epoch in range(n_epochs):
        validate_train(train_fn, 'Train', epoch, train, batch_size)
        validate(val_fn, '\t Dev', epoch, dev, batch_size)
        validate(val_fn, '\t Test', epoch, test, batch_size)
        print "\n"
