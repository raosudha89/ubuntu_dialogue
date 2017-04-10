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

def iterate_minibatches(contexts, responses, labels, batch_size, shuffle=False):
    if shuffle:
        indices = np.arange(len(contexts))
        np.random.shuffle(indices)
    for start_idx in range(0, len(contexts) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield contexts[excerpt], responses[excerpt], labels[excerpt]

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
            no_of_turns = np.bincount(content)[EOT_IDX]
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

def validate_old(val_fn, fold_name, epoch, fold, batch_size):
    start = time.time()
    num_batches = 0.
    cost = 0.
    acc = 0.
    contexts, contexts_feat_vec, responses, responses_feat_vec = fold
    #ignore features
    labels = np.zeros((len(contexts), 2), dtype=np.int32)
    for i in range(len(contexts)):
        labels[i][0] = 1
    for c, r, l in iterate_minibatches(contexts, responses, labels, \
                                                batch_size, shuffle=True):

        cm, cus, cum, cup, cfv = get_utt_masks(c)
        r = r[:,0]
        rm, rus, rum, rup, rfv = get_utt_masks(r, is_response=True)        
        #pdb.set_trace()
        r_2 = r[:,1]
        rm_2, rus_2, rum_2, rup_2, rfv_2 = get_utt_masks(r_2, is_response=True)
        
        loss, probs = val_fn(c, cm, cus, cum, cup, cfv, r, rm, rus, rum, rup, rfv, l)
        corr = 0
        for i in range(len(probs)):
            if np.argmax(probs[i]) == np.argmax(l[i]):
                corr += 1
        acc += corr*1.0/len(probs)    
        cost += loss*1.0/len(probs)
        num_batches += 1
    lstring = '%s: epoch:%d, cost:%f, acc:%f, time:%d' % \
                (fold_name, epoch, cost / num_batches, acc / num_batches, time.time()-start)
    print lstring

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
    for c, cm, r, rm in iterate_minibatches(contexts, context_masks, responses_list, response_masks_list, \
                                                batch_size, shuffle=True):
        l = np.zeros((batch_size, N), dtype=np.int32)
        l[:,0] = 1
        r, rm, l = shuffle(r, rm, l)
        probs = np.zeros((batch_size, N))
        for j in range(N):
            out = val_fn(c, cm, r[:,j,:], rm[:,j,:], l[:,j])
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

def get_lstm_output(layers, d_hidden):
    l_context_emb, l_context_mask, l_response_emb, l_response_mask = layers 
    # now feed sequences of spans into VAN
    
    l_context_lstm = lasagne.layers.LSTMLayer(l_context_emb_feat, d_hidden, \
                                                mask_input=l_context_mask, \
                                                #only_return_final=True, \
                                                peepholes=False,\
                                                )
    
    l_response_lstm = lasagne.layers.LSTMLayer(l_response_emb_feat, d_hidden, \
                                                mask_input=l_response_mask, \
                                                #only_return_final=True, \
                                                peepholes=False,\
                                                )

    context_out = lasagne.layers.get_output(l_context_lstm)
    responses_out = lasagne.layers.get_output(l_response_lstm)
    
    context_params = lasagne.layers.get_all_params(l_context_lstm, trainable=True)
    response_params = lasagne.layers.get_all_params(l_response_lstm, trainable=True)

    return context_out, response_out, context_params+response_params

def build_lstm(len_voc, d_word, d_hidden, max_len, batch_size, lr, freeze=False):

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

    labels = T.imatrix(name='label')   
 
    # define network
    l_context = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=contexts)
    l_response = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=responses)

    l_context_emb = lasagne.layers.EmbeddingLayer(l_context, len_voc, d_word, W=lasagne.init.GlorotNormal())
    l_response_emb = lasagne.layers.EmbeddingLayer(l_response, len_voc, d_word, W=l_context_emb.W)
    
    l_context_feat = lasagne.layers.InputLayer(shape=(batch_size, max_len, FEATURE_COUNT), input_var=contexts_feat_vec)
    l_response_feat = lasagne.layers.InputLayer(shape=(batch_size, max_len, FEATURE_COUNT), input_var=responses_feat_vec)

    #concatenate emb and feat layers
    l_context_emb_feat = lasagne.layers.ConcatLayer([l_context_emb, l_context_feat], axis=2)
    l_response_emb_feat = lasagne.layers.ConcatLayer([l_response_emb, l_response_feat], axis=2)
    
    l_context_mask = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=context_masks)
    l_response_mask = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=response_masks)
    
    layers = [l_context_emb_feat, l_context_mask, l_response_emb_feat, l_response_mask]
    context_out, response_out, lstm_params = get_lstm_output(layers, d_hidden)
    
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

    context_response = T.concatenate([context_utt_out, response_utt_out], axis=1)
    l_context_response_in = lasagne.layers.InputLayer(shape=(batch_size, 2*d_hidden), input_var=context_response)

    for k in range(DEPTH):
        if k == 0:
            l_context_response_dense = lasagne.layers.DenseLayer(l_context_response_in, num_units=d_hidden, \
                                                             nonlinearity=lasagne.nonlinearities.rectify)
        else:
            l_context_response_dense = lasagne.layers.DenseLayer(l_context_response_dense, num_units=d_hidden, \
                                                             nonlinearity=lasagne.nonlinearities.rectify)
    
    l_context_response_dense = lasagne.layers.DenseLayer(l_context_response_dense, num_units=1, \
                                                             nonlinearity=lasagne.nonlinearities.sigmoid)
    
    dense_params = lasagne.layers.get_all_params(l_context_response_dense, trainable=True)
    
    probs = lasagne.layers.get_output(l_context_response_dense)        
    
    loss = T.sum(lasagne.objectives.binary_crossentropy(probs, labels))

    updates = lasagne.updates.adam(loss, lstm_params + utt_lstm_params + dense_params, learning_rate=lr)
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
        ct = np.bincount(context)[EOU_IDX] 
        MAX_UTT_LEN = max(ct, MAX_UTT_LEN)
    MAX_UTT_LEN += 1

if __name__ == '__main__':
    np.set_printoptions(linewidth=160)
    if len(sys.argv) < 4:
        print "usage: python lstm_ubuntu.py train.p dev.p vocab_size.p vocab_idx.p batch_size"
        sys.exit(0)
    train = cPickle.load(open(sys.argv[1], 'rb'))
    dev = cPickle.load(open(sys.argv[2], 'rb'))
    vocab_size = cPickle.load(open(sys.argv[3], 'rb'))
    vocab_idx = cPickle.load(open(sys.argv[4], 'rb'))
    batch_size = int(sys.argv[5])
    d_word = 100
    d_hidden = 100
    freeze = False
    lr = 0.001
    n_epochs = 20
    rho = 1e-5
    max_len = train[0].shape[1]

    vocab_size += 1
    EOU_IDX = vocab_idx['__eou__']        
    EOT_IDX = vocab_idx['__eot__']        
    print 'max_utt_len', MAX_UTT_LEN    

    print 'vocab_size', vocab_size, 'max_len', max_len

    print 'compiling graph...'
    start_time = time.time()
    train_fn, val_fn = build_lstm(vocab_size, d_word, d_hidden, max_len, batch_size, lr, rho, freeze=freeze)
    print 'done compiling'
    end_time = time.time()           
    print("--- %s seconds ---" % (end_time - start_time))

    # train network
    for epoch in range(n_epochs):
        validate(train_fn, 'Train', epoch, train, batch_size)
        validate(val_fn, '\t DEV', epoch, dev, batch_size)
        print "\n"
