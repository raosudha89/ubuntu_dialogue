#!/bin/bash

#PBS -S /bin/sh
#PBS -N create_we
#PBS -l pmem=64g
#PBS -m abe
#PBS -l walltime=10:00:00 

DATA_DIR=/fs/clip-scratch/raosudha/
SCRIPT_DIR=/fs/clip-amr/ubuntu_dialogue/src
python $SCRIPT_DIR/create_we_vocab.py $DATA_DIR/glove.840B.300d.txt $DATA_DIR/glove.840B.300d.we.p $DATA_DIR/glove.840B.300d.vocab.p
