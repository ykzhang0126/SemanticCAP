#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
#export PATH=/home/.ykzhang/env/tf22/tf2py37/bin:$PATH
export d=~/data_process
# eval "$(conda shell.bash hook)"
source ~/software/anaconda3/bin/activate tf25
# conda activate tf25
which python

seq_length=257
epoch=32
batch_size=256
lr=2e-5
#CUDA_VISIBLE_DEVICES=1
# rm -rf $d/data/output$seq_length

python $d/lm_model/train_LM.py $seq_length $epoch $batch_size post_train $lr


