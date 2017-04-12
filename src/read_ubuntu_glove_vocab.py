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

COUNT_CUTOFF = 15
tknzr = TweetTokenizer()

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

def get_idx(content, vocab_idx, UNK_idx, max_len, is_response=True):
    content_idx = np.zeros((max_len), dtype=np.int32)
    content_mask_idx = np.concatenate((np.ones(len(content)), np.zeros(max_len-len(content))))
    for i, w in enumerate(content):
        try:
            content_idx[i] = vocab_idx[w]
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

def read_data(data_file, vocab, max_context_len, max_response_len, is_train=True):
    contexts = []
    responses_list = []
    labels = []
    for line in csv.reader(data_file):
        line = [unicode(l, 'utf-8') for l in line]
        context = line[0]
        context_tok = word_tokenize(context)
        if len(context_tok) > max_context_len:
            context_tok = context_tok[-max_context_len:]
        context_tok = preprocess(context_tok)
        contexts.append(context_tok)
        if is_train:
            response_list = [line[1]]
            labels.append(int(line[2]))
        else:
            response_list = line[1:]
        response_tok_list = []
        for response in response_list:
            response_tok = word_tokenize(response)
            response_tok = response_tok[:max_response_len]
            response_tok = preprocess(response_tok)
            response_tok_list.append(response_tok)
        responses_list.append(response_tok_list)
    if is_train:
        return contexts, responses_list, labels
    else:
        return contexts, responses_list

def get_data_idx(contexts, responses_list, max_context_len, max_response_len, vocab_idx, UNK_idx):
    contexts_idx = np.zeros((len(contexts), max_context_len), dtype=np.int32)
    context_masks_idx = np.zeros((len(contexts), max_context_len), dtype=np.float32)
    responses_count = len(responses_list[0])
    responses_list_idx = np.zeros((len(responses_list), responses_count, max_response_len), dtype=np.int32)
    response_masks_list_idx = np.zeros((len(responses_list), responses_count, max_response_len), dtype=np.float32) 
    for i in range(len(contexts)):
        context_idx, context_mask_idx = get_idx(contexts[i], vocab_idx, UNK_idx,\
                                                max_context_len, False)
        contexts_idx[i] = context_idx
        context_masks_idx[i] = context_mask_idx
        for j in range(len(responses_list[i])):
            responses_list_idx[i][j], response_masks_list_idx[i][j] = get_idx(responses_list[i][j], vocab_idx, \
                                                                              UNK_idx, max_response_len, True)

    return contexts_idx, context_masks_idx, responses_list_idx, response_masks_list_idx

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print "usage: read_ubuntu.py <train.csv> <dev.csv> <max_context_len> <max_response_len> <glove_vocab> <out_train.p> <out_dev.p> <out.vocab> <out.vocab_size> <out.vocab_idx>"
        sys.exit(0)
    start_time = time.time()
    train_file = open(sys.argv[1])
    dev_file = open(sys.argv[2])
    max_context_len = int(sys.argv[3])
    max_response_len = int(sys.argv[4])
    glove_vocab = cPickle.load(open(sys.argv[5], 'rb'))
    line = train_file.readline()
    all_sentences = []
    t_contexts, t_responses_list, labels = read_data(train_file, glove_vocab, max_context_len, max_response_len, is_train=True)
    d_contexts, d_responses_list = read_data(dev_file, glove_vocab, max_context_len, max_response_len, is_train=False)
    end_time = time.time()           
    print("--- %s seconds ---" % (end_time - start_time))
    start_time = end_time
    vocab_size = len(glove_vocab)
    print("Vocab size %s" % vocab_size) 

    UNK_idx = vocab_size 
    vocab_size += 1 #for UNK token

    labels = np.asarray(labels, dtype=np.int32)
    train_contexts_idx, train_context_masks_idx, \
        train_responses_list_idx, train_response_masks_list_idx = \
                            get_data_idx(t_contexts, t_responses_list, \
                                            max_context_len, max_response_len, vocab_idx, UNK_idx)

    dev_contexts_idx, dev_context_masks_idx, \
            dev_responses_list_idx, dev_response_masks_list_idx = \
                            get_data_idx(d_contexts, d_responses_list, \
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
    cPickle.dump(train, open(sys.argv[5], 'wb'))
    cPickle.dump(dev, open(sys.argv[6], 'wb'))
    cPickle.dump(vocab, open(sys.argv[7], 'wb'))
    cPickle.dump(vocab_size, open(sys.argv[8], 'wb'))
    cPickle.dump(vocab_idx, open(sys.argv[9], 'wb'))
    end_time = time.time() 
    print("--- %s seconds ---" % (end_time - start_time))
