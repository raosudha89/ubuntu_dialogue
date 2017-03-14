SHELL=/bin/bash

data_dir=/data/t-sudrao/data/ubuntu-ranking-dataset-creator/src
c_len=100
r_len=100
set=1
dev=gpu0
batch_size=50
tok=0
twitter_we=~/GloVe/glove.twitter.27B.100d.we

ifeq ($(tok), 1)
	data_dir=~/msr_intern2016/data/ubuntu-ranking-dataset-creator/src/twttok
endif

ifeq ($(set), 0)
	train_data=$(data_dir)/mini_train
	dev_data=$(data_dir)/mini_valid
endif

ifeq ($(set), 1)
	train_data=$(data_dir)/megatrain_10K
	dev_data=$(data_dir)/valid_1K
endif

ifeq ($(set), 2)
	train_data=$(data_dir)/megatrain_100K
	dev_data=$(data_dir)/valid_10K
endif

ifeq ($(set), 3)
	train_data=$(data_dir)/megatrain
	dev_data=$(data_dir)/valid
endif

ifeq ($(set), 4)
	train_data=$(data_dir)/megatrain_100K_200K
	dev_data=$(data_dir)/valid_10K_20K
endif

ifeq ($(set), 5)
	train_data=$(data_dir)/megatrain_500K
	dev_data=$(data_dir)/valid
endif

THEANO=$(pip list | grep 'Theano (0.9.0dev1)')
LASAGNE=$(pip list | grep Lasagne)

export PATH:=$(PATH):/usr/local/cuda-8.0/bin:/usr/local/cuda-8.0/lib64
export CUDA_ROOT:=/usr/local/cuda-8.0
export LD_LIBRARY_PATH:=$(LD_LIBRARY_PATH):/usr/local/cuda-8.0/lib64

all: 
	@cat README.md

ifeq ($(THEANO), )
	pip install --user --upgrade https://github.com/Theano/Theano/archive/master.zip
endif

ifeq ($(LASAGNE), )
	pip install --user --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
endif

download.punkt:
	pip install nltk
	pip install numpy
	python -m nltk.downloader punkt

ubuntu.read: download.punkt
ubuntu.read:
	python scripts/read_ubuntu.py $(train_data).csv $(dev_data).csv $(c_len) $(r_len) $(train_data).$(c_len)_$(r_len).p $(dev_data).$(c_len)_$(r_len).p $(train_data).$(c_len)_$(r_len).vocab $(train_data).$(c_len)_$(r_len).vocab_size $(train_data).$(c_len)_$(r_len).vocab_idx

ubuntu.read.feature: download.punkt
ubuntu.read.feature:
	python scripts/read_ubuntu_feature.py $(train_data).csv $(dev_data).csv $(c_len) $(r_len) $(train_data).$(c_len)_$(r_len).feat.p $(dev_data).$(c_len)_$(r_len).feat.p $(train_data).$(c_len)_$(r_len).feat.vocab $(train_data).$(c_len)_$(r_len).feat.vocab_size $(train_data).$(c_len)_$(r_len).feat.vocab_idx

ubuntu.read.feature.bursty: download.punkt
ubuntu.read.feature.bursty:
	python scripts/read_ubuntu_feature_bursty.py $(train_data).csv $(dev_data).csv $(c_len) $(r_len) $(train_data).$(c_len)_$(r_len).feat.bursty.p $(dev_data).$(c_len)_$(r_len).feat.bursty.p $(train_data).$(c_len)_$(r_len).feat.bursty.vocab $(train_data).$(c_len)_$(r_len).feat.bursty.vocab_size $(train_data).$(c_len)_$(r_len).feat.bursty.vocab_idx

ubuntu.lstm: 
	THEANO_FLAGS=device=$(dev),floatX=float32 python scripts/lstm_ubuntu.py $(train_data).$(c_len)_$(r_len).p $(dev_data).$(c_len)_$(r_len).p $(train_data).$(c_len)_$(r_len).vocab_size $(batch_size)

#best model
ubuntu.lstm.heirarchical.speaker: 
	THEANO_FLAGS=device=$(dev),floatX=float32 python scripts/lstm_ubuntu_heirarchical_speaker.py $(train_data).$(c_len)_$(r_len).feat.p $(dev_data).$(c_len)_$(r_len).feat.p $(train_data).$(c_len)_$(r_len).feat.vocab_size $(train_data).$(c_len)_$(r_len).feat.vocab_idx $(batch_size)

ubuntu.lstm.heirarchical.speaker.bursty: 
	THEANO_FLAGS=device=$(dev),floatX=float32 python scripts/lstm_ubuntu_heirarchical_speaker.py $(train_data).$(c_len)_$(r_len).feat.bursty.p $(dev_data).$(c_len)_$(r_len).feat.bursty.p $(train_data).$(c_len)_$(r_len).feat.bursty.vocab_size $(train_data).$(c_len)_$(r_len).feat.bursty.vocab_idx $(batch_size)




ubuntu.lstm.heirarchical.speaker.cluster: 
	THEANO_FLAGS=device=$(dev),floatX=float32 python scripts/lstm_ubuntu_heirarchical_speaker_cluster.py $(train_data).$(c_len)_$(r_len).feat.p $(dev_data).$(c_len)_$(r_len).feat.p $(train_data).$(c_len)_$(r_len).feat.vocab_size $(train_data).$(c_len)_$(r_len).feat.vocab_idx $(batch_size)

ubuntu.lstm.heirarchical.speaker.utt: 
	THEANO_FLAGS=device=$(dev),floatX=float32 python scripts/lstm_ubuntu_heirarchical_speaker_utt.py $(train_data).$(c_len)_$(r_len).feat.p $(dev_data).$(c_len)_$(r_len).feat.p $(train_data).$(c_len)_$(r_len).feat.vocab_size $(train_data).$(c_len)_$(r_len).feat.vocab_idx $(batch_size)

ubuntu.lstm.heirarchical.speaker.feature: 
	THEANO_FLAGS=device=$(dev),floatX=float32 python scripts/lstm_ubuntu_heirarchical_speaker_feature.py $(train_data).$(c_len)_$(r_len).feat.p $(dev_data).$(c_len)_$(r_len).feat.p $(train_data).$(c_len)_$(r_len).feat.vocab_size $(train_data).$(c_len)_$(r_len).feat.vocab_idx $(batch_size)

ubuntu.lstm.heirarchical.speaker.utt_feature: 
	THEANO_FLAGS=device=$(dev),floatX=float32 python scripts/lstm_ubuntu_heirarchical_speaker_utt_feature.py $(train_data).$(c_len)_$(r_len).feat.p $(dev_data).$(c_len)_$(r_len).feat.p $(train_data).$(c_len)_$(r_len).feat.vocab_size $(train_data).$(c_len)_$(r_len).feat.vocab_idx $(batch_size)

ubuntu.lstm.heirarchical.attn: 
	THEANO_FLAGS=device=$(dev),floatX=float32 python scripts/lstm_ubuntu_heirarchical_attn_mask.py $(train_data).$(c_len)_$(r_len).feat.p $(dev_data).$(c_len)_$(r_len).feat.p $(train_data).$(c_len)_$(r_len).feat.vocab_size $(train_data).$(c_len)_$(r_len).feat.vocab_idx $(batch_size)

ubuntu.lstm.heirarchical.cluster: 
	THEANO_FLAGS=device=$(dev),floatX=float32 python scripts/lstm_ubuntu_heirarchical_cluster.py $(train_data).$(c_len)_$(r_len).feat.p $(dev_data).$(c_len)_$(r_len).feat.p $(train_data).$(c_len)_$(r_len).feat.vocab_size $(train_data).$(c_len)_$(r_len).feat.vocab_idx $(batch_size)

ubuntu.lstm.attn: 
	THEANO_FLAGS=device=$(dev),floatX=float32 python scripts/lstm_attn_ubuntu.py $(train_data).$(c_len)_$(r_len).p $(dev_data).$(c_len)_$(r_len).p $(train_data).$(c_len)_$(r_len).vocab_size $(batch_size)

ubuntu.lstm.attn.mask: 
	THEANO_FLAGS=device=$(dev),floatX=float32 python scripts/lstm_attn_mask_ubuntu.py $(train_data).$(c_len)_$(r_len).p $(dev_data).$(c_len)_$(r_len).p $(train_data).$(c_len)_$(r_len).vocab_size $(batch_size)

