#!/bin/bash

SECONDS=0

date 

# Main work starts

## main experiments: methods [D] and [C], 40 datasets, 3 seeds

for SEED in "21" "31" "51" 
do
	for BR in "8" "1"
	do
		for DATA in "BCT_Micro" "BRD_Micro" "CRS_Micro" "FLW_Micro" "MD_MIX_Micro" "PLK_Micro" "PLT_VIL_Micro" "RESISC_Micro" "SPT_Micro" "TEX_Micro" "ACT_40_Micro" "APL_Micro" "DOG_Micro" "INS_2_Micro" "MD_5_BIS_Micro" "MED_LF_Micro" "PLT_NET_Micro" "PNU_Micro" "RSICB_Micro" "TEX_DTD_Micro" "ACT_410_Micro" "AWA_Micro" "BTS_Micro" "FNG_Micro" "INS_Micro" "MD_6_Micro" "PLT_DOC_Micro" "PRT_Micro" "RSD_Micro" "TEX_ALOT_Micro" "ARC_Micro" "ASL_ALP_Micro" "BFY_Micro" "BRK_Micro" "MD_5_T_Micro" "PLT_LVS_Micro" "POT_TUB_Micro" "SNK_Micro" "UCMLU_Micro" "VEG_FRU_Micro"
		do
			python run.py --nb_blocks 1,1,1,1 --epochs 300 --ema_epochs 60 --dataset ${DATA} --split_pretrained_weights --nb_branch ${BR} --random_seed ${SEED} --pretrained_model resnet152 --divergence_idx 15 --warm_stem_epochs 10 --img_size 128 --loss cross_entropy --weight_decay 0.01 --num_workers 4 --lr 0.001 --batch_size 32 --optimizer AdamW
		done
	done
done

## main experiments: methods [A] and [B], 40 datasets, 3 seeds

for SEED in "21" "31" "51" 
do
	for TR in "only_last" "retrain_all"
	do
		for DATA in "BCT_Micro" "BRD_Micro" "CRS_Micro" "FLW_Micro" "MD_MIX_Micro" "PLK_Micro" "PLT_VIL_Micro" "RESISC_Micro" "SPT_Micro" "TEX_Micro" "ACT_40_Micro" "APL_Micro" "DOG_Micro" "INS_2_Micro" "MD_5_BIS_Micro" "MED_LF_Micro" "PLT_NET_Micro" "PNU_Micro" "RSICB_Micro" "TEX_DTD_Micro" "ACT_410_Micro" "AWA_Micro" "BTS_Micro" "FNG_Micro" "INS_Micro" "MD_6_Micro" "PLT_DOC_Micro" "PRT_Micro" "RSD_Micro" "TEX_ALOT_Micro" "ARC_Micro" "ASL_ALP_Micro" "BFY_Micro" "BRK_Micro" "MD_5_T_Micro" "PLT_LVS_Micro" "POT_TUB_Micro" "SNK_Micro" "UCMLU_Micro" "VEG_FRU_Micro"
		do
			python run.py --run_baseline --baseline_model resnet152 --train_layers ${TR} --nb_blocks 3,8,36,3 --nb_branch 1 --epochs 300 --ema_epochs 60 --dataset ${DATA} --split_pretrained_weights --random_seed ${SEED} --pretrained_model resnet152 --img_size 128 --loss cross_entropy --weight_decay 0.01 --num_workers 4 --lr 0.001 --batch_size 32 --optimizer AdamW
		done
	done
done

# Main work ends

DURATION=$SECONDS

echo "End of the program! $(($DURATION / 60)) minutes and $(($DURATION % 60)) seconds elapsed." 
