SHELL=/bin/bash

glove_data_dir=/fs/clip-scratch/raosudha
data_dir=/fs/clip-amr/ubuntu_dialogue/ubuntu-ranking-dataset-creator/src
c_len=100
r_len=100
set=1
dev=gpu0
batch_size=500

ifeq ($(set), 0)
	train_data=$(data_dir)/mini_train
	dev_data=$(data_dir)/mini_valid
endif

ifeq ($(set), 1)
	train_data=$(data_dir)/train
	dev_data=$(data_dir)/valid
endif

all: 
	@cat README.md

download.punkt:
	python -m nltk.downloader punkt
	python -m nltk.downloader averaged_perceptron_tagger

ubuntu.read: download.punkt
ubuntu.read:
	python src/read_ubuntu.py $(train_data).csv $(dev_data).csv $(c_len) $(r_len) $(train_data).$(c_len)_$(r_len).p $(dev_data).$(c_len)_$(r_len).p $(train_data).$(c_len)_$(r_len).vocab $(train_data).$(c_len)_$(r_len).vocab_size $(train_data).$(c_len)_$(r_len).vocab_idx

ubuntu.read.feature: download.punkt
ubuntu.read.feature:
	python src/read_ubuntu_feature.py $(train_data).csv $(dev_data).csv $(c_len) $(r_len) $(train_data).$(c_len)_$(r_len).feat.p $(dev_data).$(c_len)_$(r_len).feat.p $(train_data).$(c_len)_$(r_len).feat.vocab $(train_data).$(c_len)_$(r_len).feat.vocab_size $(train_data).$(c_len)_$(r_len).feat.vocab_idx

ubuntu.read.bursty: download.punkt
ubuntu.read.bursty:
	python src/read_ubuntu_bursty.py $(train_data).csv $(dev_data).csv $(c_len) $(r_len) $(train_data).$(c_len)_$(r_len).bursty.p $(dev_data).$(c_len)_$(r_len).bursty.p $(train_data).$(c_len)_$(r_len).bursty.vocab $(train_data).$(c_len)_$(r_len).bursty.vocab_size $(train_data).$(c_len)_$(r_len).bursty.vocab_idx

ubuntu.lstm: 
	THEANO_FLAGS=device=$(dev),floatX=float32 python src/lstm_ubuntu.py $(train_data).$(c_len)_$(r_len).p $(dev_data).$(c_len)_$(r_len).p $(train_data).$(c_len)_$(r_len).vocab_idx $(train_data).$(c_len)_$(r_len).vocab_size $(batch_size) $(glove_data_dir)/glove.840B.300d.we.p $(glove_data_dir)/glove.840B.300d.vocab.p

ubuntu.lstm.bursty: 
	THEANO_FLAGS=device=$(dev),floatX=float32 python src/lstm_ubuntu.py $(train_data).$(c_len)_$(r_len).bursty.p $(dev_data).$(c_len)_$(r_len).bursty.p $(train_data).$(c_len)_$(r_len).bursty.vocab_idx $(train_data).$(c_len)_$(r_len).bursty.vocab_size $(batch_size) $(glove_data_dir)/glove.840B.300d.we.p $(glove_data_dir)/glove.840B.300d.vocab.p

#best model
ubuntu.lstm.heirarchical.speaker: 
	THEANO_FLAGS=device=$(dev),floatX=float32 python src/lstm_ubuntu_heirarchical_speaker.py $(train_data).$(c_len)_$(r_len).p $(dev_data).$(c_len)_$(r_len).p $(train_data).$(c_len)_$(r_len).vocab_idx $(train_data).$(c_len)_$(r_len).vocab_size $(batch_size) $(glove_data_dir)/glove.840B.300d.we.p $(glove_data_dir)/glove.840B.300d.vocab.p

ubuntu.lstm.heirarchical.speaker.bursty: 
	THEANO_FLAGS=device=$(dev),floatX=float32 python src/lstm_ubuntu_heirarchical_speaker.py $(train_data).$(c_len)_$(r_len).feat.bursty.p $(dev_data).$(c_len)_$(r_len).feat.bursty.p $(train_data).$(c_len)_$(r_len).feat.bursty.vocab_size $(train_data).$(c_len)_$(r_len).feat.bursty.vocab_idx $(batch_size)




ubuntu.lstm.heirarchical.speaker.cluster: 
	THEANO_FLAGS=device=$(dev),floatX=float32 python src/lstm_ubuntu_heirarchical_speaker_cluster.py $(train_data).$(c_len)_$(r_len).feat.p $(dev_data).$(c_len)_$(r_len).feat.p $(train_data).$(c_len)_$(r_len).feat.vocab_size $(train_data).$(c_len)_$(r_len).feat.vocab_idx $(batch_size)

ubuntu.lstm.heirarchical.speaker.utt: 
	THEANO_FLAGS=device=$(dev),floatX=float32 python src/lstm_ubuntu_heirarchical_speaker_utt.py $(train_data).$(c_len)_$(r_len).feat.p $(dev_data).$(c_len)_$(r_len).feat.p $(train_data).$(c_len)_$(r_len).feat.vocab_size $(train_data).$(c_len)_$(r_len).feat.vocab_idx $(batch_size)

ubuntu.lstm.heirarchical.speaker.feature: 
	THEANO_FLAGS=device=$(dev),floatX=float32 python src/lstm_ubuntu_heirarchical_speaker_feature.py $(train_data).$(c_len)_$(r_len).feat.p $(dev_data).$(c_len)_$(r_len).feat.p $(train_data).$(c_len)_$(r_len).feat.vocab_size $(train_data).$(c_len)_$(r_len).feat.vocab_idx $(batch_size)

ubuntu.lstm.heirarchical.speaker.utt_feature: 
	THEANO_FLAGS=device=$(dev),floatX=float32 python src/lstm_ubuntu_heirarchical_speaker_utt_feature.py $(train_data).$(c_len)_$(r_len).feat.p $(dev_data).$(c_len)_$(r_len).feat.p $(train_data).$(c_len)_$(r_len).feat.vocab_size $(train_data).$(c_len)_$(r_len).feat.vocab_idx $(batch_size)

ubuntu.lstm.heirarchical.attn: 
	THEANO_FLAGS=device=$(dev),floatX=float32 python src/lstm_ubuntu_heirarchical_attn_mask.py $(train_data).$(c_len)_$(r_len).feat.p $(dev_data).$(c_len)_$(r_len).feat.p $(train_data).$(c_len)_$(r_len).feat.vocab_size $(train_data).$(c_len)_$(r_len).feat.vocab_idx $(batch_size)

ubuntu.lstm.heirarchical.cluster: 
	THEANO_FLAGS=device=$(dev),floatX=float32 python src/lstm_ubuntu_heirarchical_cluster.py $(train_data).$(c_len)_$(r_len).feat.p $(dev_data).$(c_len)_$(r_len).feat.p $(train_data).$(c_len)_$(r_len).feat.vocab_size $(train_data).$(c_len)_$(r_len).feat.vocab_idx $(batch_size)

ubuntu.lstm.attn: 
	THEANO_FLAGS=device=$(dev),floatX=float32 python src/lstm_attn_ubuntu.py $(train_data).$(c_len)_$(r_len).p $(dev_data).$(c_len)_$(r_len).p $(train_data).$(c_len)_$(r_len).vocab_size $(batch_size)

ubuntu.lstm.attn.mask: 
	THEANO_FLAGS=device=$(dev),floatX=float32 python src/lstm_attn_mask_ubuntu.py $(train_data).$(c_len)_$(r_len).p $(dev_data).$(c_len)_$(r_len).p $(train_data).$(c_len)_$(r_len).vocab_size $(batch_size)

