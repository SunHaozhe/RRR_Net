import os
import time
import shutil
import gc
import itertools

import sklearn
import torch

import wandb

import millify

from thop import profile

from sub_resnet152 import ResNet152Helper, SubResNet152, get_diverged_resnet152_models

from utils import *


def count_all_parameters(model):
    return sum([x.numel() for x in model.parameters()])


def reorder_columns(df):
    def remove_prefix(x):
        if x.startswith(prefixes[0]):
            prefix = prefixes[0]
            x = x[len(prefix):]
        elif x.startswith(prefixes[1]):
            prefix = prefixes[1]
            x = x[len(prefix):]
        elif x.startswith(prefixes[2]):
            prefix = prefixes[2]
            x = x[len(prefix):]
        return x, prefix

    prefixes = ["flops_", "nb_params_"]

    column_names = df_stem.columns.tolist()
    column_names = [xx for xx in column_names if not xx.endswith("_h")]

    part_str_prefix = [remove_prefix(xx)for xx in column_names]

    sort_tuples = []
    for idx, (part_str, prefix) in enumerate(part_str_prefix):
        nb_branch = int(part_str.split("_")[0])
        column_name = column_names[idx]
        sort_tuples.append((column_name, prefix, nb_branch))

    sort_tuples = sorted(sort_tuples, key=lambda x: (x[1], x[2]))

    column_names = [xx[0] for xx in sort_tuples]

    column_names_h = ["{}_h".format(xx) for xx in column_names]

    column_names = column_names_h + column_names

    df = df.reindex(columns=column_names)

    return df


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    
    # split resnet152
    parser.add_argument("--divergence_idx", type=int, default=15)
    parser.add_argument("--split_pretrained_weights",
                        action="store_true", default=False)

    args = parser.parse_args()
    return args


def compute_FLOPs_one_model(model, input_shape, device):
    x = torch.randn(input_shape).to(device)

    input_sizes = []
    for idx, module in enumerate(model.blocks):
        input_sizes.append(tuple(x.shape))
        x = module(x)

    flops_per_block = []

    for idx, module in enumerate(model.blocks):
        input = torch.randn(input_sizes[idx]).to(device)
        macs, params = profile(module, inputs=(input,))
        flops_per_block.append(macs)

    return flops_per_block


def compute_stem_branch_flops_per_block(stem, branch, input_shape, device):
    x = torch.randn(input_shape).to(device)
    stem_input_size = tuple(x.shape)
    x = stem(x)
    branch_input_size = tuple(x.shape)

    print("stem_input_size={}, branch_input_size={}".format(
        stem_input_size, branch_input_size))

    stem_flops_per_block = compute_FLOPs_one_model(
        stem, stem_input_size, device)
    branch_flops_per_block = compute_FLOPs_one_model(
        branch, branch_input_size, device)
    return stem_flops_per_block, branch_flops_per_block


def compute_stem_branch_nb_params_per_block(stem, branch):
    stem_nb_params_per_block = []
    for idx, module in enumerate(stem.blocks):
        stem_nb_params_per_block.append(count_trainable_parameters(module))
        
    branch_nb_params_per_block = []
    for idx, module in enumerate(branch.blocks):
        branch_nb_params_per_block.append(count_trainable_parameters(module))
        
    return stem_nb_params_per_block, branch_nb_params_per_block


if __name__ == "__main__":

    args = parse_arguments()

    batch_size = 1
    img_size = 128

    nb_classes = 20

    input_shape = (batch_size, 3, img_size, img_size)
    
    device = get_device()
    
    nb_branch_list = [1, 8]

    df_stem = []
    df_branch = []
    for nb_branch in nb_branch_list:

        base_models = get_diverged_resnet152_models(divergence_idx=args.divergence_idx,
                                                    nb_branch=nb_branch,
                                                    nb_classes=nb_classes,
                                                    split_pretrained_weights=args.split_pretrained_weights,
                                                    device=device, concat=True,
                                                    train_dataloader=None,
                                                    warm_stem_epochs=0,
                                                    only_first=True)
        
        model = base_models[0]

        del base_models
        gc.collect()

        stem = model.stem
        branch = model.branch

        stem_flops_per_block, \
            branch_flops_per_block = compute_stem_branch_flops_per_block(
            stem, branch, input_shape, device)
        
        stem_nb_params_per_block, \
            branch_nb_params_per_block = compute_stem_branch_nb_params_per_block(
            stem, branch)

        df_stem.append(stem_nb_params_per_block)
        df_branch.append(branch_nb_params_per_block)
        
        df_stem.append(stem_flops_per_block)
        df_branch.append(branch_flops_per_block)
        

    column_names = [["nb_params_{}".format(xx), 
                     "flops_{}".format(xx)] \
                         for xx in nb_branch_list]
    
    # flatten a list of lists into a flat list
    column_names = list(itertools.chain(*column_names))

    df_stem = pd.DataFrame(df_stem).T
    df_stem.columns = column_names

    df_branch = pd.DataFrame(df_branch).T
    df_branch.columns = column_names

    for column_name in column_names:
        df_stem["{}_h".format(column_name)] = df_stem[column_name].apply(
            lambda x: millify.millify(x, precision=2)
        )
        df_branch["{}_h".format(column_name)] = df_branch[column_name].apply(
            lambda x: millify.millify(x, precision=2)
        )
    
    df_stem.to_csv("log_flops_params_stem.csv")
    df_branch.to_csv("log_flops_params_branch.csv")

    print(df_stem)
    print()
    print(df_branch)
