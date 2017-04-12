import sys
import theano, lasagne, cPickle, time                                                   
import numpy as np
import theano.tensor as T     
from collections import OrderedDict, Counter
import math, random
import pdb

EOU_IDX = -1
EOT_IDX = -1
MAX_UTT_LEN = 23
FEATURE_COUNT = 1
DEPTH = 1
N = 10

def iterate_minibatches_train(contexts, responses, labels, batch_size, shuffle=False):
    if shuffle:
        indices = np.arange(len(contexts))
        np.random.shuffle(indices)
    for start_idx in range(0, len(contexts) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield contexts[excerpt], responses[excerpt], labels[excerpt]
        
def iterate_minibatches(contexts, responses, batch_size, shuffle=False):
    if shuffle:
        indices = np.arange(len(contexts))
        np.random.shuffle(indices)
    for start_idx in range(0, len(contexts) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield contexts[excerpt], responses[excerpt]

def get_utt_masks(contents, is_response=False):
    contents_masks = np.zeros(contents.shape, dtype=np.float32)
    contents_utt_splits = np.zeros((contents.shape[0], MAX_UTT_LEN, contents.shape[1]), dtype=np.float32)
    contents_utt_masks = np.zeros((contents.shape[0], MAX_UTT_LEN), dtype=np.float32)
    contents_utt_prods = np.zeros((contents.shape[0], MAX_UTT_LEN), dtype=np.float32)
    contents_feat_vec = np.zeros((contents.shape[0], contents.shape[1], FEATURE_COUNT), dtype=np.float32)
    for i, content in enumerate(contents):
        prev = 0
        k = 0
        #for response: speaker_id is always 0, so don't change contents_feat_vec
        if not is_response:
            #for context: speaker_id starts with 1 if odd #of turns, 0 if even
            try:
                no_of_turns = np.bincount(content)[EOT_IDX]
            except:
                print 'content empty'
                no_of_turns = 0
            speaker_id = 1
            if no_of_turns%2 == 0:
                speaker_id = 0

        mask_set = False
        for j, idx in enumerate(content):
            if not is_response:
                contents_feat_vec[i][j][0] = speaker_id
                if idx == EOT_IDX:
                    speaker_id = 1 - speaker_id #switch the speaker_id    
            if idx == EOU_IDX:
                contents_utt_splits[i][k][prev:j+1] = 1.0/(j+1-prev)
                prev = j+1
                k += 1
            elif idx == 0: #content has ended
                contents_masks[i][:j] = 1
                mask_set = True
                contents_utt_splits[i][k][prev:j+1] = 1.0/(j+1-prev)
                k += 1
                break
        if not mask_set: #content is of max_len
            contents_masks[i] = 1
        contents_utt_masks[i][:k] = 1
        if k == 0:
            contents_utt_prods[i][:k] = 1.0
        else:
            contents_utt_prods[i][:k] = 1.0/k
    return contents_masks, contents_utt_splits, contents_utt_masks, contents_utt_prods, contents_feat_vec

def shuffle(r, l):
    shuffled_r = np.zeros(r.shape, dtype=np.int32)
    shuffled_l = np.zeros(l.shape, dtype=np.int32)

    for i in range(len(r)):
        indexes = range(len(r[i]))
        random.shuffle(indexes)
        for j, index in enumerate(indexes):
            shuffled_r[i][j] = r[i][index]
            shuffled_l[i][j] = l[i][index]

    return shuffled_r, shuffled_l

def validate_train(val_fn, fold_name, epoch, fold, batch_size):
    start = time.time()
    num_batches = 0.
    cost = 0.
    acc = 0.
    contexts, context_masks, responses, response_masks, labels = fold
    responses = responses[:,0,:]
    response_masks = response_masks[:,0,:]
    for c, r, l in iterate_minibatches_train(contexts, responses, labels, batch_size, shuffle=True):
        cm, cus, cum, cup, cfv = get_utt_masks(c)
        rm, rus, rum, rup, rfv = get_utt_masks(r, is_response=True)
        loss, probs = val_fn(c, cm, cus, cum, cup, cfv, r, rm, rus, rum, rup, rfv, l)
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
    for c, r in iterate_minibatches(contexts, responses_list, batch_size, shuffle=True):
        cm, cus, cum, cup, cfv = get_utt_masks(c)
        l = np.zeros((batch_size, N), dtype=np.int32)
        l[:,0] = 1
        r, l = shuffle(r, l)
        probs = np.zeros((batch_size, N))
        for j in range(N):
            rm, rus, rum, rup, rfv = get_utt_masks(r[:,j,:], is_response=True)
            out = val_fn(c, cm, cus, cum, cup, cfv, r[:,j,:], rm, rus, rum, rup, rfv, l[:,j])
            loss = out[0]
            probs[:,j] = out[1][:,0]
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

def get_lstm_output(layers, d_hidden, forget_gate_bias):
    l_context_emb, l_context_mask, l_response_emb, l_response_mask = layers 
    # now feed sequences of spans into VAN
    
    l_context_lstm = lasagne.layers.LSTMLayer(l_context_emb, d_hidden, \
                                            grad_clipping=10, \
                                            forgetgate=lasagne.layers.Gate(b=lasagne.init.Constant(forget_gate_bias)), \
                                            mask_input=l_context_mask, \
                                            learn_init=True, \
                                            peepholes=True,\
                                            )
 
    l_response_lstm = lasagne.layers.LSTMLayer(l_response_emb, d_hidden, \
                                            grad_clipping=10, \
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

    context_out = lasagne.layers.get_output(l_context_lstm)
    response_out = lasagne.layers.get_output(l_response_lstm)
    
    context_params = lasagne.layers.get_all_params(l_context_lstm, trainable=True)

    return context_out, response_out, context_params

def build_lstm(len_voc, d_word, d_hidden, max_len, batch_size, lr, rho, forget_gate_bias, freeze=False):

    # input theano vars
    contexts = T.imatrix(name='context')
    context_masks = T.matrix(name='context_mask')
    context_utt_splits = T.ftensor3(name='context_utt_split')
    context_utt_masks = T.matrix(name='context_utt_mask')
    context_utt_prods = T.matrix(name='context_utt_prod')
    contexts_feat_vec = T.ftensor3(name='context_feat')

    responses = T.imatrix(name='response')
    response_masks = T.matrix(name='response_mask')
    response_utt_splits = T.ftensor3(name='response_utt_split')
    response_utt_masks = T.matrix(name='response_utt_mask')
    response_utt_prods = T.matrix(name='response_utt_prod')
    responses_feat_vec = T.ftensor3(name='response_feat')

    labels = T.ivector(name='label')   
 
    # define network
    l_context = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=contexts)
    l_response = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=responses)

    l_context_emb = lasagne.layers.EmbeddingLayer(l_context, len_voc, d_word, W=word_embeddings)
    l_response_emb = lasagne.layers.EmbeddingLayer(l_response, len_voc, d_word, W=l_context_emb.W)
    
    l_context_feat = lasagne.layers.InputLayer(shape=(batch_size, max_len, FEATURE_COUNT), input_var=contexts_feat_vec)
    l_response_feat = lasagne.layers.InputLayer(shape=(batch_size, max_len, FEATURE_COUNT), input_var=responses_feat_vec)

    #concatenate emb and feat layers
    l_context_emb_feat = lasagne.layers.ConcatLayer([l_context_emb, l_context_feat], axis=2)
    l_response_emb_feat = lasagne.layers.ConcatLayer([l_response_emb, l_response_feat], axis=2)
    
    l_context_mask = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=context_masks)
    l_response_mask = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=response_masks)
    
    layers = [l_context_emb_feat, l_context_mask, l_response_emb_feat, l_response_mask]
    context_out, response_out, lstm_params = get_lstm_output(layers, d_hidden, forget_gate_bias)
    
    context_utt_emb = T.sum(context_utt_splits[:,:,:,None] * context_out[:,None,:,:] , axis=2)
    response_utt_emb = T.sum(response_utt_splits[:,:,:,None] * response_out[:,None,:,:] , axis=2)

    d_emb = d_word    
    l_context_utt_emb = lasagne.layers.InputLayer(shape=(batch_size, MAX_UTT_LEN, d_emb), input_var=context_utt_emb) 
    l_context_utt_mask = lasagne.layers.InputLayer(shape=(batch_size, MAX_UTT_LEN), input_var=context_utt_masks)
    l_response_utt_emb = lasagne.layers.InputLayer(shape=(batch_size, MAX_UTT_LEN, d_emb), input_var=response_utt_emb) 
    l_response_utt_mask = lasagne.layers.InputLayer(shape=(batch_size, MAX_UTT_LEN), input_var=response_utt_masks)

    layers = [l_context_utt_emb, l_context_utt_mask, l_response_utt_emb, l_response_utt_mask]
    context_utt_out, response_utt_out, utt_lstm_params = get_lstm_output(layers, d_hidden)

    context_utt_out = T.sum(context_utt_out * context_utt_prods[:,:,None], axis=1)
    response_utt_out = T.sum(response_utt_out * response_utt_prods[:,:,None], axis=1)

    # context_response = T.concatenate([context_utt_out, response_utt_out], axis=1)
    # l_context_response_in = lasagne.layers.InputLayer(shape=(batch_size, 2*d_hidden), input_var=context_response)
    # 
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
    # dense_params = lasagne.layers.get_all_params(l_context_response_dense, trainable=True)
    # probs = lasagne.layers.get_output(l_context_response_dense)

    M = theano.shared(np.eye(d_hidden, dtype=np.float32))

    probs = T.sum(T.dot(context_utt_out, M)*response_utt_out, axis=1)
    probs = lasagne.nonlinearities.sigmoid(probs)
    
    loss = T.sum(lasagne.objectives.binary_crossentropy(probs, labels))
    
    # all_params = lstm_params + utt_lstm_params + dense_params
    all_params = lstm_params + utt_lstm_params + [M]
    loss += rho * sum(T.sum(l ** 2) for l in all_params)

    updates = lasagne.updates.adam(loss, all_params, learning_rate=lr)
    train_fn = theano.function([contexts, context_masks, context_utt_splits, context_utt_masks, context_utt_prods, contexts_feat_vec, \
                                responses, response_masks, response_utt_splits, response_utt_masks, response_utt_prods, responses_feat_vec, labels], \
                               [loss, probs], updates=updates)
    val_fn = theano.function([contexts, context_masks, context_utt_splits, context_utt_masks, context_utt_prods, contexts_feat_vec, \
                                responses, response_masks, response_utt_splits, response_utt_masks, response_utt_prods, responses_feat_vec, labels], \
                               [loss, probs])
    return train_fn, val_fn

def set_max_utt_len(contexts):
    global MAX_UTT_LEN
    for context in contexts:
        try:
            ct = np.bincount(context)[EOU_IDX]
        except:
            ct = 0
        MAX_UTT_LEN = max(ct, MAX_UTT_LEN)
    MAX_UTT_LEN += 1

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
        print "usage: python lstm_ubuntu.py train.p dev.p vocab_size.p vocab_idx.p batch_size glove_we glove_vocab"
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
    n_epochs = 5
    rho = 1e-5
    forget_gate_bias = 2.0
    max_len = train[0].shape[1]

    word_embeddings = get_word_embeddings(vocab_idx, vocab_size, glove_we, glove_vocab, d_word)
    word_embeddings = np.asarray(word_embeddings, dtype=np.float32)

    EOU_IDX = vocab_idx['__eou__']        
    EOT_IDX = vocab_idx['__eot__']
    set_max_utt_len(train[0])
    print 'max_utt_len', MAX_UTT_LEN    

    print 'vocab_size', vocab_size, 'max_len', max_len

    print 'compiling graph...'
    start_time = time.time()
    train_fn, val_fn = build_lstm(vocab_size, d_word, d_hidden, max_len, batch_size, lr, rho, forget_gate_bias, freeze=freeze)
    #train_fn, val_fn = None, None
    print 'done compiling'
    end_time = time.time()           
    print("--- %s seconds ---" % (end_time - start_time))

    # train network
    for epoch in range(n_epochs):
        validate_train(train_fn, 'Train', epoch, train, batch_size)
        validate(val_fn, '\t DEV', epoch, dev, batch_size)
        print "\n"
