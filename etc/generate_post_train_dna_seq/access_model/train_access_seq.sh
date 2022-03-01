#!/bin/bash
export d=~/data_process
# source ~/software/anaconda3/bin/activate tf25

max_len=256
batch_size=8
epoch=64
seq_len=64
save_memory=1

#python initial_model_weights.py $seq_len

access_dir=$d/dna_access/
if [ $save_memory -eq 1 ]
then
    access_dir=${access_dir}/train_with_pre-pretrained-date/
fi
python ${access_dir}/train_access.py $seq_len $max_len $epoch $batch_size

