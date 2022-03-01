#!/bin/bash
#module load apps/bedtools/2.29.1/gcc-7.3.1
d=~/user/ykzhang/data_process
bed_file=$d/generate_random_dna_seq/dna_rand_extract.bed

touch $bed_file
echo -n "" > $bed_file
for i in `seq 1 22`
do
	for j in `seq 1 5000`	# 4e7
	do
		st=$((j*8000))
		ed=$((st+1600))
		echo -e "chr$i\t$st\t$ed" >> $bed_file
	done
done

bedtools getfasta -fi $d/data/pre_train/hg19.fa -bed $bed_file  -fo $d/data/pre_train/dna_rand.fa
#module rm apps/bedtools/2.29.1/gcc-7.3.1
