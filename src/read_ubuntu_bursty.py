import sys
import numpy as np
import nltk
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
CONTEXT_COUNT_CUTOFF = 2
pos_dict = {}
tknzr = TweetTokenizer()
MAX_UTT_LEN = 23

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

def get_idx(content, content_pos, vocab_idx, max_len, UNK_idx):
    content_idx = np.zeros((max_len), dtype=np.int32)
    content_mask_idx = np.concatenate((np.ones(len(content)), np.zeros(max_len-len(content))))
    for i, w in enumerate(content):
        try:
            content_idx[i] = vocab_idx[w]
        except KeyError:
            try:
                content_idx[i] = pos_dict[content_pos[i][1]]
            except KeyError:
                content_idx[i] = UNK_idx
    return content_idx, content_mask_idx

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

def read_data(data_file, max_context_len, max_response_len, vocab=None, context_vocab=None, is_train=False):
    contexts = []
    responses_list = []
    contexts_pos = []
    responses_pos_list = []
    labels = []
    for line in csv.reader(data_file):
        line = [unicode(l, 'utf-8') for l in line]
        context = line[0]
        context_tok = word_tokenize(context)
        if len(context_tok) > max_context_len:
            context_tok = context_tok[-max_context_len:]
        context_tok = preprocess(context_tok)
        context_pos = nltk.pos_tag(context_tok)
        if is_train:
            context_word_freq = {}
            for w in context_tok:
                vocab[w] += 1
                try:
                    context_word_freq[w] += 1
                except KeyError:
                    context_word_freq[w] = 1
            for w, ct in context_word_freq.iteritems():
                try:
                    context_vocab[w] = max(context_vocab[w], ct)
                except KeyError:
                    context_vocab[w] = ct
        contexts.append(context_tok)
        contexts_pos.append(context_pos)
        if is_train:
            response_list = [line[1]]
            labels.append(int(line[2]))
        else:
            response_list = line[1:]
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
        return contexts, responses_list, contexts_pos, responses_pos_list, labels, vocab, context_vocab
    else:
        return contexts, responses_list, contexts_pos, responses_pos_list

def get_data_idx(contexts, responses_list, contexts_pos, responses_pos_list, max_context_len, max_response_len, vocab_idx, UNK_idx):
    contexts_idx = np.zeros((len(contexts), max_context_len), dtype=np.int32)
    context_masks_idx = np.zeros((len(contexts), max_context_len), dtype=np.float32)
    responses_count = len(responses_list[0])
    responses_list_idx = np.zeros((len(responses_list), responses_count, max_response_len), dtype=np.int32)
    response_masks_list_idx = np.zeros((len(responses_list), responses_count, max_response_len), dtype=np.float32) 
    for i in range(len(contexts)):
        context_idx, context_mask_idx = get_idx(contexts[i], contexts_pos[i], \
                                                vocab_idx, max_context_len, UNK_idx)
        contexts_idx[i] = context_idx
        context_masks_idx[i] = context_mask_idx
        for j in range(len(responses_list[i])):
            responses_list_idx[i][j], response_masks_list_idx[i][j]  = get_idx(responses_list[i][j], responses_pos_list[i][j], \
                                                                                vocab_idx, max_response_len, UNK_idx)

    return contexts_idx, context_masks_idx, responses_list_idx, response_masks_list_idx

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print "usage: read_ubuntu.py <train.csv> <dev.csv> <test.csv> <out_train.p> <out_dev.p> <out.vocab> <out.vocab_size> <out.vocab_idx>"
        sys.exit(0)
    start_time = time.time()
    train_file = open(sys.argv[1])
    dev_file = open(sys.argv[2])
    test_file = open(sys.argv[3])
    max_context_len = int(sys.argv[4])
    max_response_len = int(sys.argv[5])
    line = train_file.readline()
    all_sentences = []
    vocab_size = 0
    vocab = collections.defaultdict(int)
    context_vocab = collections.defaultdict(int)
    t_contexts, t_responses_list, t_contexts_pos, t_responses_list_pos, labels, vocab, context_vocab = read_data(train_file, max_context_len, max_response_len, vocab, context_vocab, is_train=True)
    d_contexts, d_responses_list, d_contexts_pos, d_responses_list_pos = read_data(dev_file, max_context_len, max_response_len)
    tt_contexts, tt_responses_list, tt_contexts_pos, tt_responses_list_pos = read_data(test_file, max_context_len, max_response_len)
    
    end_time = time.time()           
    print("--- %s seconds ---" % (end_time - start_time))
    start_time = end_time
    print("Entire vocab size %s" % len(vocab)) 
    vocab = collections.OrderedDict(sorted(vocab.items(), key=lambda t: t[1], reverse=True))
    vocab = {w:ct for w,ct in vocab.iteritems() if ct > COUNT_CUTOFF}
    print("Cutoff vocab size %s" % len(vocab)) 

    print("Entire context vocab size %s" % len(context_vocab))
    context_vocab = collections.OrderedDict(sorted(context_vocab.items(), key=lambda t: t[1], reverse=True))
    context_vocab = {w:ct for w,ct in context_vocab.iteritems() if ct > CONTEXT_COUNT_CUTOFF}
    print("Cutoff context vocab size %s" % len(context_vocab)) 

    vocab = context_vocab

    vocab_dim = 100
    vocab_size = len(vocab.keys())
    tags = load('help/tagsets/upenn_tagset.pickle').keys()
    for i in range(len(tags)):
        pos_dict[tags[i]] = i + 1 + vocab_size
    vocab_size += len(tags)
    vocab_size += 1 #for UNK token
    UNK_idx = 1
    idx = 2 # 0 is for UNK
    vocab_idx = {}
    for w, ct in vocab.iteritems():
        vocab_idx[w] = idx
        idx += 1
    print("Vocab size used %s" % vocab_size)
    end_time = time.time()           
    print("--- %s seconds ---" % (end_time - start_time))
    start_time = end_time

    labels = np.asarray(labels, dtype=np.int32)
    train_contexts_idx, train_context_masks_idx, \
        train_responses_list_idx, train_response_masks_list_idx = \
                            get_data_idx(t_contexts, t_responses_list, t_contexts_pos, t_responses_list_pos, \
                                            max_context_len, max_response_len, vocab_idx, UNK_idx)

    dev_contexts_idx, dev_context_masks_idx, \
            dev_responses_list_idx, dev_response_masks_list_idx = \
                            get_data_idx(d_contexts, d_responses_list, d_contexts_pos, d_responses_list_pos, \
                                            max_context_len, max_response_len, vocab_idx, UNK_idx)

    test_contexts_idx, test_context_masks_idx, \
            test_responses_list_idx, test_response_masks_list_idx = \
                            get_data_idx(tt_contexts, tt_responses_list, tt_contexts_pos, tt_responses_list_pos, \
                                            max_context_len, max_response_len, vocab_idx, UNK_idx)

    end_time = time.time()           
    print("--- %s seconds ---" % (end_time - start_time))
    start_time = end_time

    end_time = time.time()           
    print("--- %s seconds ---" % (end_time - start_time))
    start_time = end_time
    train = [train_contexts_idx, train_context_masks_idx,\
            train_responses_list_idx, train_response_masks_list_idx, labels]
    dev = [dev_contexts_idx, dev_context_masks_idx, \
             dev_responses_list_idx, dev_response_masks_list_idx]
    test = [test_contexts_idx, test_context_masks_idx, \
             test_responses_list_idx, test_response_masks_list_idx]
    cPickle.dump(train, open(sys.argv[6], 'wb'))
    cPickle.dump(dev, open(sys.argv[7], 'wb'))
    cPickle.dump(test, open(sys.argv[8], 'wb'))
    cPickle.dump(vocab, open(sys.argv[9], 'wb'))
    cPickle.dump(vocab_size, open(sys.argv[10], 'wb'))
    cPickle.dump(vocab_idx, open(sys.argv[11], 'wb'))
    end_time = time.time() 
    print("--- %s seconds ---" % (end_time - start_time))
