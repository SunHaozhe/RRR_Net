#!/bin/bash

SECONDS=0


date 


# Main work starts

## select the best "number of branches" on the held-out dataset: icdar03_char_micro

for SEED in "21" "31" "41" "51" "61"
do
	for DATA in "icdar03_char_micro" 
	do
		for BR in "1" "2" "4" "8" "16" "32" "64" "128"     
		do
			python run.py --nb_blocks 1,1,1,1 --epochs 300 --ema_epochs 60 --do_validation --dataset ${DATA} --split_pretrained_weights --nb_branch ${BR} --random_seed ${SEED} --pretrained_model resnet152 --divergence_idx 15 --warm_stem_epochs 10 --img_size 128 --loss cross_entropy --weight_decay 0.01 --num_workers 4 --lr 0.001 --batch_size 32 --optimizer AdamW
		done
	done
done

# Main work ends

DURATION=$SECONDS

echo "End of the program! $(($DURATION / 60)) minutes and $(($DURATION % 60)) seconds elapsed." 
