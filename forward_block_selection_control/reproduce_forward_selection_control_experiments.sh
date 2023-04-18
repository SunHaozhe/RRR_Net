#!/bin/bash


SECONDS=0


date 

# Main work starts

## Control experiment of the forward block selection experiment: 
## train ResNet_1_1_1_1 for 47*300=14100 epochs. 
## 5 runs, each with a different seed. 

for SEED in "21" "31" "41" "51" "61" 
do
	for BLOCKS in "--nb_blocks 1,1,1,1" 
	do
		python run.py --paper_res --wandb_project_name PickResnet152Blocks --use_micro_version --use_albumentations --use_ImageNet_pretraining --num_workers 4 --epochs 14100 ${BLOCKS} --random_seed ${SEED} --batch_size 32 --loss cross_entropy --img_size 128 --img_mode RGB
	done
done



# Main work ends

DURATION=$SECONDS

echo "End of the program! $(($DURATION / 60)) minutes and $(($DURATION % 60)) seconds elapsed." 