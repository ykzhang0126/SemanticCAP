#!/bin/bash
export d=~/data_process
source ~/software/anaconda3/bin/activate tf25

python $d/dna_access/preprocess.py $d/data/fine_tune/SNEDE0000EMT_pos $d/data/fine_tune/SNEDE0000EMT_neg

python $d/dna_access/2_code_seq.py $d/data/fine_tune/SNEDE0000EMT

max_len=256
batch_num=3000
seq_len=768

rm -rf $d/data/fine_tune/input$max_len
mkdir -p $d/data/fine_tune/input$max_len/tmp


python $d/dna_access/3_split_seq.py $max_len $batch_num
python $d/dna_access/merge_input.py $max_len $seq_len
