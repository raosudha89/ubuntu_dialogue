import sys
import numpy as np
import nltk
#from gensim.models import word2vec
#from gensim.models import Word2Vec
import cPickle
import csv
import time
import collections
from nltk.data import load
from nltk.tokenize import TweetTokenizer
import re
import itertools
import pdb
import cPickle as p
from math import log

COUNT_CUTOFF = 15
pos_dict = {}
tknzr = TweetTokenizer()
FEATURE_COUNT = 10
MAX_UTT_LEN = 23
#MAX_VOCAB_SIZE = 14000
UNK_IDX = 1

FLAGS = re.MULTILINE | re.DOTALL

def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = "<hashtag> {} <allcaps>".format(hashtag_body)
    else:
        result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
    return result

def allcaps(text):
    text = text.group()
    return text.lower() + " <allcaps>"

def replace_emoticons(text):
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        try:
            return re.sub(pattern, repl, text, flags=FLAGS)
        except:
            return text

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"/"," / ")
    text = re_sub(r"@\w+", "<user>")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
    text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
    text = re_sub(r"<3","<heart>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    text = re_sub(r"#\S+", hashtag)
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")

    ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    text = re_sub(r"([A-Z]){2,}", allcaps)

    return text.lower()

def contains(mylist, candidates):
	for c in candidates:
		if c in mylist:
			return True
	return False

def is_sublist(mylist, sublist):
	if " ".join(sublist) in " ".join(mylist):
		return True
	return False

def get_feature_vector(content):
	turns = mysplit(content, '__eot__')
	
	# question - 1-feature value per utterance & broadcast feature value to all tokens in utterance
	# emoticon use - per turn
	# ack - ok, okay, yeah, yes, agree, sounds&good - per utterance
	# is_first_person_pro - I,me,my,we - per utterance
	# is_sec_person_pro - you - per utterance
	# is_third_person_pro - it,they,he,she,him
	# yes - yea, yeah, yes, agree, right
	# no - no, not, 
	
	# Illocutionary acts - per utterance
	# 	is_directive - 'can you', 'could you', 'would you', 'will you', 'please', 'you should',
	#	is_commissive - 'i can', 'i will', 'i shall', 'i would', 'i will', 'i should',
	#	is_expressive - (same as emoticon use above)
	#	is_assertive - if none of above

	# *greeting - 'thank', 'ty', 'thnx'


	fv = np.zeros((MAX_UTT_LEN, FEATURE_COUNT), dtype=np.float32)

	questions = ['?', 'how', 'what', 'why', 'who', 'when', 'where', 'whom', 'which']
	ack = ['ok', 'okay', 'agree']
	first_pro = ['i', 'me', 'my', 'we']
	second_pro = ['you']
	third_pro = ['it', 'he', 'she', 'they']
	yes_term = ['yea', 'yes', 'yeah', 'agree', 'right']
	no_term = ['no', 'not']

	is_directive = ['can you', 'could you', 'would you', 'will you', 'please', 'you should']
	is_commissive = ['i can', 'i will', 'i shall', 'i would', 'i will', 'i should']
	is_expressive = ['<smile>', '<lolface>', '<neutralface>', '<heart>', '<sadface>']

	for i, turn in enumerate(turns):
		utterances = mysplit(turn, '__eou__')
		#isQuestion feature
		for j, utterance in enumerate(utterances):			
			#question feature
			if contains(utterance, questions):
				fv[j][0] = 1
			#agreement feature
			if contains(utterance, ack):
				fv[j][1] = 1
			#pronoun feature
			if contains(utterance, first_pro):
				fv[j][2] = 1 
			if contains(utterance, second_pro):
				fv[j][3] = 1 
			if contains(utterance, third_pro):
				fv[j][4] = 1  
			#yes/no feature
			if contains(utterance, yes_term):
				fv[j][5] = 1 
			if contains(utterance, no_term):
				fv[j][6] = 1 
			#Illocutionary acts
			if is_sublist(utterance, is_directive):
				fv[j][7] = 1
			elif is_sublist(utterance, is_commissive):
				fv[j][8] = 1
			elif contains(utterance, is_expressive):
				fv[j][9] = 1

	return fv

def get_idx(content, content_pos, vocab_idx, max_len, is_context=False, content_word_freq=None, vocab_doc_ct=None, N=0):
	content_idx = np.zeros((max_len), dtype=np.float32)
	for i, w in enumerate(content):
		#if is_context:
		#	tfidf = content_word_freq[w]*log(N*1.0/vocab_doc_ct[w])
		#	if tfidf < TFIDF_CUTOFF:
		#		try:
		#			content_idx[i] = pos_dict[content_pos[i][1]]
		#		except KeyError:
		#			content_idx[i] = UNK_IDX
		#		continue

		try:
			content_idx[i] = vocab_idx[w]
		except KeyError:
			try:
				content_idx[i] = pos_dict[content_pos[i][1]]
			except KeyError:
				content_idx[i] = UNK_IDX
	return content_idx

def preprocess(words):
	#words = [w.lower()[:7] for w in words]
	words = [w.lower() for w in words]
	return words

def word_tokenize(sent):
	#return nltk.word_tokenize(sent)

	#words = re.split('\W+|_', sent)
	#words = [w for w in words if w]
	#return words
	
	sent = replace_emoticons(sent)
	return tknzr.tokenize(sent)

def mysplit(mylist, tok):
	ret_lists = []
	l = []
	for v in mylist:
		l.append(v)
		if v == tok:
			if l:
				ret_lists.append(l)
			l = []
	if l:
		ret_lists.append(l)
	return ret_lists

def read_data(data_file, vocab, vocab_doc_ct, max_context_len, max_response_len, is_train=True):
	contexts = []
	responses_list = []
	contexts_pos = []
	responses_pos_list = []
	contexts_word_freq = []
	for line in csv.reader(data_file):
		line = [unicode(l, 'utf-8') for l in line]
		#for l in line:
		#	print l
		#pdb.set_trace()
		context = line[0]
		context_tok = word_tokenize(context)
		if len(context_tok) > max_context_len:
			context_tok = context_tok[-max_context_len:]
		context_tok = preprocess(context_tok)
		context_pos = nltk.pos_tag(context_tok)
		context_word_freq = {}
		for w in context_tok:
			if is_train:
				vocab[w] += 1
			try:
				context_word_freq[w] += 1
			except KeyError:
				context_word_freq[w] = 1
				vocab_doc_ct[w] += 1
		contexts.append(context_tok)
		contexts_pos.append(context_pos)
		contexts_word_freq.append(context_word_freq)
		response_list = line[1:3]
		response_tok_list = []
		response_pos_list = []
		for response in response_list:
			response_tok = word_tokenize(response)
			response_tok = response_tok[:max_response_len]
			response_tok = preprocess(response_tok)
			response_pos = nltk.pos_tag(response_tok)
			if is_train:
				for w in response_tok:
					vocab[w] += 1
			response_tok_list.append(response_tok)
			response_pos_list.append(response_pos)
		
		responses_list.append(response_tok_list)
		responses_pos_list.append(response_pos_list)
	if is_train:
		return contexts, responses_list, contexts_pos, responses_pos_list, contexts_word_freq, vocab, vocab_doc_ct
	else:
		return contexts, responses_list, contexts_pos, responses_pos_list, contexts_word_freq, vocab_doc_ct

def get_data_idx(contexts, responses_list, contexts_pos, responses_pos_list, contexts_word_freq, max_context_len, max_response_len, vocab_idx, vocab_doc_ct):
	contexts_idx = np.zeros((len(contexts), max_context_len), dtype=np.int32)
	contexts_feat_vec = np.zeros((len(contexts), MAX_UTT_LEN, FEATURE_COUNT), dtype=np.float32)
	responses_count = len(responses_list[0])
	responses_list_idx = np.zeros((len(responses_list), responses_count, max_response_len), dtype=np.int32)
	responses_list_feat_vec = np.zeros((len(responses_list), responses_count, MAX_UTT_LEN, FEATURE_COUNT), dtype=np.float32)
	for i in range(len(contexts)):
		context_idx = get_idx(contexts[i], contexts_pos[i], \
								vocab_idx, max_context_len, True, \
								contexts_word_freq[i], vocab_doc_ct, len(contexts))
		contexts_feat_vec[i] = get_feature_vector(contexts[i])
		contexts_idx[i] = context_idx
		for j in range(len(responses_list[i])):
			responses_list_idx[i][j] = get_idx(responses_list[i][j], responses_pos_list[i][j], \
												vocab_idx, max_response_len, False)
			responses_list_feat_vec[i][j] = get_feature_vector(responses_list[i][j])

	return contexts_idx, contexts_feat_vec, responses_list_idx, responses_list_feat_vec

def create_vocab_idx(contexts_word_freq, vocab_doc_ct):
	vocab_idx = {}
	idx = 2 # 0 is for _not_a_word_, 1 is for UNK
	N = len(contexts_word_freq)
	max_word_freq = collections.defaultdict(int)
	for context_word_freq in contexts_word_freq:
		for w in context_word_freq.keys():
			max_word_freq[w] = max(context_word_freq[w], max_word_freq[w])
	
	TFIDF_CUTOFF = log(N*1.0)
	for w in max_word_freq.keys():
		tfidf = max_word_freq[w]*log(N*1.0/vocab_doc_ct[w])
		if tfidf > TFIDF_CUTOFF:
			vocab_idx[w] = idx
			idx += 1
		#else:
		#	if w not in ['__eou__', '__eot__']:
		#		print w
	try:
		v = vocab_idx['__eou__']
	except KeyError:
		vocab_idx['__eou__'] = idx
		idx += 1
	try:
		v = vocab_idx['__eot__']
	except KeyError:
		vocab_idx['__eot__'] = idx

	return vocab_idx

if __name__ == "__main__":
	if len(sys.argv) < 4:
		print "usage: read_ubuntu.py <train.csv> <dev.csv> <out_train.p> <out_dev.p> <out.vocab> <out.vocab_size> <out.vocab_idx>"
		sys.exit(0)
	start_time = time.time()
	train_file = open(sys.argv[1])
	dev_file = open(sys.argv[2])
	max_context_len = int(sys.argv[3])
	max_response_len = int(sys.argv[4])
	line = train_file.readline()
	all_sentences = []
	vocab_size = 0
	vocab = collections.defaultdict(int)
	vocab_doc_ct = collections.defaultdict(int)
	t_contexts, t_responses_list, t_contexts_pos, t_responses_list_pos, t_contexts_word_freq, vocab, vocab_doc_ct = read_data(train_file, vocab, vocab_doc_ct, max_context_len, max_response_len, is_train=True)
	d_contexts, d_responses_list, d_contexts_pos, d_responses_list_pos, d_contexts_word_freq, vocab_doc_ct = read_data(dev_file, vocab, vocab_doc_ct, max_context_len, max_response_len, is_train=False)
	
	end_time = time.time()           
	print("--- %s seconds ---" % (end_time - start_time))
	start_time = end_time
	print("Entire Vocab size %s" % len(vocab)) 
	#vocab = collections.OrderedDict(sorted(vocab.items(), key=lambda t: t[1], reverse=True)[:MAX_VOCAB_SIZE])
	vocab = collections.OrderedDict(sorted(vocab.items(), key=lambda t: t[1], reverse=True))
	#vocab = {w:ct for w,ct in vocab.iteritems() if ct > COUNT_CUTOFF}

	vocab_idx = create_vocab_idx(t_contexts_word_freq+d_contexts_word_freq, vocab_doc_ct)
	vocab_size = len(vocab_idx.keys()) + 2
	print("Vocab size used %s" % vocab_size)
	tags = load('help/tagsets/upenn_tagset.pickle').keys()
	for i in range(len(tags)):
		pos_dict[tags[i]] = i + 1 + vocab_size
	vocab_size += len(tags)
	
	end_time = time.time()           
	print("--- %s seconds ---" % (end_time - start_time))
	start_time = end_time

	train_contexts_idx, train_contexts_feat_vec, \
		train_responses_list_idx, train_responses_list_feat_vec = \
							get_data_idx(t_contexts, t_responses_list, t_contexts_pos, t_responses_list_pos, t_contexts_word_freq, \
											max_context_len, max_response_len, vocab_idx, vocab_doc_ct)

	dev_contexts_idx, dev_contexts_feat_vec, \
			dev_responses_list_idx, dev_responses_list_feat_vec = \
							get_data_idx(d_contexts, d_responses_list, d_contexts_pos, d_responses_list_pos, d_contexts_word_freq, \
											max_context_len, max_response_len, vocab_idx, vocab_doc_ct)

	end_time = time.time()           
	print("--- %s seconds ---" % (end_time - start_time))
	start_time = end_time

	end_time = time.time()           
	print("--- %s seconds ---" % (end_time - start_time))
	start_time = end_time
	train = [train_contexts_idx, train_contexts_feat_vec, \
			train_responses_list_idx, train_responses_list_feat_vec]
	dev = [dev_contexts_idx, dev_contexts_feat_vec, \
         	dev_responses_list_idx, dev_responses_list_feat_vec]
	cPickle.dump(train, open(sys.argv[5], 'wb'))
	cPickle.dump(dev, open(sys.argv[6], 'wb'))
	cPickle.dump(vocab, open(sys.argv[7], 'wb'))
	cPickle.dump(vocab_size, open(sys.argv[8], 'wb'))
	cPickle.dump(vocab_idx, open(sys.argv[9], 'wb'))
	end_time = time.time()           
	print("--- %s seconds ---" % (end_time - start_time))

