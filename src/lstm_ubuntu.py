import sys
import theano, lasagne, cPickle, time												   
import numpy as np
import theano.tensor as T	 
from collections import OrderedDict, Counter
import math, random
import pdb

N = 10
DEPTH = 0

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

def validate(val_fn, fold_name, epoch, fold, batch_size):
	start = time.time()
	num_batches = 0.
	cost = 0.
	acc = 0.
	contexts, context_masks, responses, response_masks = fold
	for c, cm, r, rm in iterate_minibatches(contexts, context_masks, responses, response_masks, \
												batch_size, shuffle=True):
		cm = cm.astype('float32')
		l = np.zeros((batch_size, N), dtype=np.int32)
		l[:,0] = 1
		r, rm, l = shuffle(r, rm, l)
		r = np.transpose(np.array(r), (1, 0, 2))
		rm = np.transpose(np.array(rm, dtype=np.float32), (1, 0, 2))
		loss, probs = val_fn(c, cm, r, rm, l)
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

def build_lstm(len_voc, d_word, d_hidden, max_len, batch_size, lr, freeze=False):

	# input theano vars
	contexts = T.imatrix(name='context')
	context_masks = T.matrix(name='context_mask')
	responses = T.itensor3(name='responses')
	response_masks = T.tensor3(name='response_masks')
	labels = T.imatrix(name='label')   
 
	# define network
	l_context = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=contexts)
	l_context_mask = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=context_masks)
	l_context_emb = lasagne.layers.EmbeddingLayer(l_context, len_voc, d_word, W=lasagne.init.GlorotNormal())
	
	l_context_lstm = lasagne.layers.LSTMLayer(l_context_emb, d_hidden, \
									mask_input=l_context_mask, \
									#only_return_final=True, \
									peepholes=False,\
									)

	l_responses = [None]*N
	l_response_masks = [None]*N
	l_response_embs = [None]*N
	l_response_lstm = [None]*N
	
	for i in range(N):
		l_responses[i] = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=responses[i])
		l_response_masks[i] = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=response_masks[i])
		l_response_embs[i] = lasagne.layers.EmbeddingLayer(l_responses[i], len_voc, d_word, W=lasagne.init.GlorotNormal())
	
	l_response_lstm[0] = lasagne.layers.LSTMLayer(l_response_embs[0], d_hidden, \
									mask_input=l_response_masks[0], \
									#only_return_final=True, \
									peepholes=False,\
									)	
	
	# now feed sequences of spans into VAN
	for i in range(1, N):
		l_response_lstm[i] = lasagne.layers.LSTMLayer(l_response_embs[i], d_hidden, \
									mask_input=l_response_masks[i], \
									#only_return_final=True, \
									ingate=lasagne.layers.Gate(W_in=l_response_lstm[0].W_in_to_ingate,\
																W_hid=l_response_lstm[0].W_hid_to_ingate,\
																b=l_response_lstm[0].b_ingate,\
																nonlinearity=l_response_lstm[0].nonlinearity_ingate),\
									outgate=lasagne.layers.Gate(W_in=l_response_lstm[0].W_in_to_outgate,\
																W_hid=l_response_lstm[0].W_hid_to_outgate,\
																b=l_response_lstm[0].b_outgate,\
																nonlinearity=l_response_lstm[0].nonlinearity_outgate),\
									forgetgate=lasagne.layers.Gate(W_in=l_response_lstm[0].W_in_to_forgetgate,\
																W_hid=l_response_lstm[0].W_hid_to_forgetgate,\
																b=l_response_lstm[0].b_forgetgate,\
																nonlinearity=l_response_lstm[0].nonlinearity_forgetgate),\
									cell=lasagne.layers.Gate(W_in=l_response_lstm[0].W_in_to_cell,\
																W_hid=l_response_lstm[0].W_hid_to_cell,\
																b=l_response_lstm[0].b_cell,\
																nonlinearity=l_response_lstm[0].nonlinearity_cell),\
									peepholes=False,\
									)

	# now get aggregate embeddings
	context_out = lasagne.layers.get_output(l_context_lstm)
	context_out = T.mean(context_out * context_masks[:,:,None], axis=1)

	responses_out = [None]*N	
	for i in range(N):
		responses_out[i] = lasagne.layers.get_output(l_response_lstm[i])
		responses_out[i] = T.mean(responses_out[i] * response_masks[i][:,:,None], axis=1)
	
	# objective computation
	M = theano.shared(np.eye(d_hidden, dtype=np.float32))

	probs = [None]*N
	for i in range(N):
		probs[i] = T.sum(T.dot(context_out, M)*responses_out[i], axis=1)
	
	probs = lasagne.nonlinearities.softmax(T.stack(probs, axis=1))
	loss = T.sum(lasagne.objectives.categorical_crossentropy(probs, labels))

	emb_context_params = lasagne.layers.get_all_params(l_context_emb, trainable=True)
	emb_response_params = lasagne.layers.get_all_params(l_response_embs[0], trainable=True)

	all_context_params = lasagne.layers.get_all_params(l_context_lstm, trainable=True)
	all_response_params = lasagne.layers.get_all_params(l_response_lstm[0], trainable=True)
	
	updates = lasagne.updates.adam(loss, emb_context_params + all_context_params + [M] + \
								   emb_response_params + all_response_params, learning_rate=lr)
	train_fn = theano.function([contexts, context_masks, responses, response_masks, labels], \
							   [loss, probs], updates=updates)
	val_fn = theano.function([contexts, context_masks, responses, response_masks, labels], \
							   [loss, probs])
	return train_fn, val_fn

if __name__ == '__main__':
	np.set_printoptions(linewidth=160)
	if len(sys.argv) < 4:
		print "usage: python lstm_ubuntu.py train.p dev.p vocab.p We.p batch_size"
		sys.exit(0)
	train = cPickle.load(open(sys.argv[1], 'rb'))
	dev = cPickle.load(open(sys.argv[2], 'rb'))
	pdb.set_trace()
	vocab_size = cPickle.load(open(sys.argv[3], 'rb'))
	batch_size = int(sys.argv[4])
	d_word = 100
	d_hidden = 100
	freeze = False
	lr = 0.001
	n_epochs = 20
	max_len = len(train[0][1])
	N = 10
	
	print 'vocab_size', vocab_size, 'max_len', max_len

	start_time = time.time()
	print 'compiling graph...'
	train_fn, val_fn = build_lstm(vocab_size, d_word, d_hidden, max_len, batch_size, lr, freeze=freeze)
	print 'done compiling'
	print 'time taken ', time.time() - start_time

	#N = int(len(train[0])*0.9)
	#train_90 = [train[0][:N], train[1][:N], train[2][:N], train[3][:N]]
	#train_10 = [train[0][N:], train[1][N:], train[2][N:], train[3][N:]]

	#M = int(len(dev[0])*0.9)
	#dev_90 = [dev[0][:M], dev[1][:M], dev[2][:M], dev[3][:M]]
	#dev_10 = [dev[0][M:], dev[1][M:], dev[2][M:], dev[3][M:]]

	# train network
	for epoch in range(n_epochs):

		validate(train_fn, 'Train', epoch, train, batch_size)
		#validate(val_fn, '\t Test on Train', epoch, train, batch_size)
		#validate(train_fn, 'TRAIN on train_90 ', epoch, train_90, batch_size) 
		#validate(train_fn, 'TRAIN on dev_90 ', epoch, dev_90, batch_size) 
		#validate(train_fn, 'Train on DEV', epoch, dev, batch_size)
		validate(val_fn, '\t DEV', epoch, dev, batch_size)
		#validate(val_fn, '\t \t TEST on train_10 ', epoch, train_10, batch_size)
		#validate(val_fn, '\t \t TEST on dev_10 ', epoch, dev_10, batch_size)
		print "\n"
