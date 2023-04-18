#!/bin/bash


SECONDS=0


date 

# Main work starts

## forward block selection v1 (5 runs, each with a different seed).
## One run means sequentially running run.py using the argument --nb_blocks from 1,1,1,1 to 3,8,36,3.


for SEED in "21" "31" "41" "51" "61" 
do
	for BLOCKS in "1,1,1,1" "2,1,1,1" "3,1,1,1" "3,2,1,1" "3,3,1,1" "3,4,1,1" "3,5,1,1" "3,6,1,1" "3,7,1,1" "3,8,1,1" "3,8,2,1" "3,8,3,1" "3,8,4,1" "3,8,5,1" "3,8,6,1" "3,8,7,1" "3,8,8,1" "3,8,9,1" "3,8,10,1" "3,8,11,1" "3,8,12,1" "3,8,13,1" "3,8,14,1" "3,8,15,1" "3,8,16,1" "3,8,17,1" "3,8,18,1" "3,8,19,1" "3,8,20,1" "3,8,21,1" "3,8,22,1" "3,8,23,1" "3,8,24,1" "3,8,25,1" "3,8,26,1" "3,8,27,1" "3,8,28,1" "3,8,29,1" "3,8,30,1" "3,8,31,1" "3,8,32,1" "3,8,33,1" "3,8,34,1" "3,8,35,1" "3,8,36,1" "3,8,36,2" "3,8,36,3"
	do
		python run.py --paper_res --use_micro_version --use_albumentations --use_ImageNet_pretraining --num_workers 4 --epochs 300 --nb_blocks ${BLOCKS} --random_seed ${SEED} --batch_size 32 --loss cross_entropy --img_size 128 --img_mode RGB
	done
done


# Main work ends

DURATION=$SECONDS

echo "End of the program! $(($DURATION / 60)) minutes and $(($DURATION % 60)) seconds elapsed." 