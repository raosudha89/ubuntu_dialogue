#!/bin/bash

#PBS -S /bin/sh
#PBS -N read_bursty
#PBS -l pmem=32g
#PBS -m abe
#PBS -l walltime=10:00:00 

source /fs/clip-amr/isi-internship/theano-env/bin/activate
cd /fs/clip-amr/ubuntu_dialogue
make ubuntu.read.bursty > ubuntu_read_bursty_withtest.out
