import sys
import theano, lasagne, cPickle, time                                                   
import numpy as np
import theano.tensor as T     
from collections import OrderedDict, Counter
import math, random
import pdb

N = 2
DEPTH = 1

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

def validate(val_fn, fold_name, epoch, fold, batch_size):
    start = time.time()
    num_batches = 0.
    cost = 0.
    acc = 0.
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
            if np.argmax(probs[i]) == np.argmax(l[i]):
                corr += 1
        acc += corr*1.0/batch_size
        num_batches += 1
    lstring = '%s: epoch:%d, cost:%f, acc:%f, time:%d' % \
                (fold_name, epoch, cost / num_batches, acc / num_batches, time.time()-start)
    print lstring

class SoftmaxLayer(lasagne.layers.Layer):
    def __init__(self, incoming, d_hid, num_units, **kwargs):
        super(SoftmaxLayer, self).__init__(incoming, **kwargs)
        self.num_units = num_units
        self.W = self.add_param(lasagne.init.GlorotUniform(), \
                (d_hid, num_units), name="W")
        self.b = self.add_param(lasagne.init.Constant(0.),
                (num_units,), name="b", regularizbale=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1])

    def get_output_for(self, input, **kwargs):
        activation = T.dot(input, self.W) + self.b
        return T.nnet.sigmoid(activation)

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
    # l_response_emb = lasagne.layers.EmbeddingLayer(l_response, len_voc, d_word, W=word_embeddings)
    # l_response_emb = lasagne.layers.EmbeddingLayer(l_response, len_voc, d_word, W=lasagne.init.GlorotNormal())
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

    # attention over embeddings
    # l_context_lstm = SoftmaxLayer(l_context_lstm, d_hidden, d_hidden)    
    # l_response_lstm = SoftmaxLayer(l_response_lstm, d_hidden, d_hidden)

    # now get aggregate embeddings
    context_out = lasagne.layers.get_output(l_context_lstm)
    context_out = T.mean(context_out * context_masks[:,:,None], axis=1)
    
    responses_out = lasagne.layers.get_output(l_response_lstm)
    responses_out = T.mean(responses_out * response_masks[:,:,None], axis=1)
    
    # objective computation
    M = theano.shared(np.eye(d_hidden, dtype=np.float32))

    probs = T.sum(T.dot(context_out, M)*responses_out, axis=1)
    probs = lasagne.nonlinearities.sigmoid(probs)

    # context_response = T.concatenate([context_out, responses_out], axis=1)
    # l_context_response_in = lasagne.layers.InputLayer(shape=(batch_size, 2*d_hidden), input_var=context_response)
    
    # for k in range(DEPTH):
    #     if k == 0:
    #         l_context_response_dense = lasagne.layers.DenseLayer(l_context_response_in, num_units=d_hidden, \
    #                                                          nonlinearity=lasagne.nonlinearities.rectify)
    #     else:
    #         l_context_response_dense = lasagne.layers.DenseLayer(l_context_response_dense, num_units=d_hidden, \
    #                                                          nonlinearity=lasagne.nonlinearities.rectify)
    # 
    # l_context_response_dense = lasagne.layers.DenseLayer(l_context_response_dense, num_units=1, \
    #                                                          nonlinearity=lasagne.nonlinearities.sigmoid)
    # 
    # probs = lasagne.layers.get_output(l_context_response_dense)        
    
    loss = T.sum(lasagne.objectives.binary_crossentropy(probs, labels))

    all_context_params = lasagne.layers.get_all_params(l_context_lstm, trainable=True)
    all_response_params = lasagne.layers.get_all_params(l_response_lstm, trainable=True)
    # dense_params = lasagne.layers.get_all_params(l_context_response_dense, trainable=True)
    # all_params =  all_context_params + all_response_params + dense_params
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
    train = cPickle.load(open(sys.argv[1], 'rb'))
    dev = cPickle.load(open(sys.argv[2], 'rb'))
    vocab_idx = cPickle.load(open(sys.argv[3], 'rb'))
    vocab_size = cPickle.load(open(sys.argv[4], 'rb'))
    batch_size = int(sys.argv[5])
    glove_we = cPickle.load(open(sys.argv[6], 'rb'))
    glove_vocab = cPickle.load(open(sys.argv[7], 'rb'))
    d_word = 300
    d_hidden = 300
    freeze = False
    lr = 0.001
    n_epochs = 10
    rho = 1e-5
    forget_gate_bias = 2.0
    max_len = len(train[0][1])
    
    word_embeddings = get_word_embeddings(vocab_idx, vocab_size, glove_we, glove_vocab, d_word)
    word_embeddings = np.asarray(word_embeddings, dtype=np.float32)
    #word_embeddings = None
    
    print 'vocab_size', vocab_size, 'max_len', max_len

    start_time = time.time()
    print 'compiling graph...'
    train_fn, val_fn = build_lstm(word_embeddings, vocab_size, d_word, d_hidden, max_len, batch_size, lr, rho, forget_gate_bias, freeze=freeze)
    print 'done compiling'
    print 'time taken ', time.time() - start_time

    # train network
    for epoch in range(n_epochs):

        validate_train(train_fn, 'Train', epoch, train, batch_size)
        validate(val_fn, '\t Dev', epoch, dev, batch_size)
        print "\n"
