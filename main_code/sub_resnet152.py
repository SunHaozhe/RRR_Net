import os
import warnings
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

import timm
from timm.models.resnet import Bottleneck, downsample_conv
from timm.models.layers.adaptive_avgmax_pool import SelectAdaptivePool2d



def count_trainable_parameters(model):
    return sum([x.numel() for x in model.parameters() if x.requires_grad])


def get_device(verbose=True):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print("Using GPU for PyTorch: {}".format(
                torch.cuda.get_device_name(
                    torch.cuda.current_device())))
    else:
        device = torch.device("cpu")
        if verbose:
            print("Using CPU for PyTorch")
    return device


class ResNet152_(nn.Module):
    """
    This class is not used in practice, it is created for sanity 
    check: whether the architecture is identical to timm's one. 
    
    Also serves as a reference.
    """
    block_kwargs = {"reduce_first": 1, "dilation": 1, "drop_block": None, "cardinality": 1,
                    "base_width": 64, "act_layer": nn.ReLU, "norm_layer": nn.BatchNorm2d, 
                    "aa_layer": None}
    down_kwargs = {'kernel_size': 1,
                   'dilation': 1, 'first_dilation': 1, 'norm_layer': nn.BatchNorm2d}
    
    def __init__(self, nb_classes=10):
        super().__init__()
        
        self.block_fn = Bottleneck
        self.inplanes = 64
        
        self.nb_blocks = [3, 8, 36, 3]
        self.planes_per_block = [64, 128, 256, 512]
        
        # conv1 part
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1,
                               affine=True, track_running_stats=True)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        
        
        
        self.layer1 = self.make_stage(0, stride=1)
        self.layer2 = self.make_stage(1, stride=2)
        self.layer3 = self.make_stage(2, stride=2)
        self.layer4 = self.make_stage(3, stride=2)
        
        self.global_pool = SelectAdaptivePool2d(pool_type="avg", flatten=True)
        self.fc = nn.Linear(512 * self.block_fn.expansion,
                            nb_classes, bias=True)
        
    def make_stage(self, stage_idx, stride):
        planes = self.planes_per_block[stage_idx]
        blocks = []
        for block_idx in range(self.nb_blocks[stage_idx]):
            
            if block_idx == 0:
                # downsampling with stride=2 with the first block
                stride = stride # 2
                downsample = downsample_conv(
                    in_channels=self.inplanes, 
                    out_channels=planes * self.block_fn.expansion, 
                    stride=stride, **self.down_kwargs)
            else:
                stride = 1
                downsample = None

            blocks.append(self.block_fn(
                self.inplanes, planes, stride, downsample, first_dilation=1,
                drop_path=None, **self.block_kwargs))
            
            self.inplanes = planes * self.block_fn.expansion
        
        stage = nn.Sequential(*blocks)
        
        return stage
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.global_pool(x)
        x = self.fc(x)
        
        return x


def verify_equivalence(model, model2):
    assert count_trainable_parameters(
        model) == count_trainable_parameters(model2)
    assert model.state_dict().keys() == model2.state_dict().keys()
    model2.load_state_dict(model.state_dict())

    model_params_shape = dict()
    for k, v in model.named_parameters():
        model_params_shape[k] = v.shape
    model2_params_shape = dict()
    for k, v in model2.named_parameters():
        model2_params_shape[k] = v.shape
    assert model_params_shape == model2_params_shape

    batch_size = 2
    input_size = 32
    dummy_input = torch.randn(batch_size, 3, input_size, input_size)
    with torch.no_grad():
        output = model(dummy_input)
        output2 = model2(dummy_input)
        assert output.shape == output2.shape

    print("verify_equivalence successfully executed!")


class ConcatenatedModel(nn.Module) :
    """
    train()/eval() overridden, they only affect branch
    
    if one wants to set training mode to both stem and 
    branch, then call train_both_stem_branch() or 
    eval_both_stem_branch()
    """
    def __init__(self, stem, branch) -> None:
        super().__init__()
        self.stem = stem
        self.branch = branch
        self.inherit_attributes()

    def forward(self, x):
        x = self.stem(x)
        x = self.branch(x)
        return x
    
    def train(self, mode=True):
        self.stem.eval()
        self.branch.train()
        
    def eval(self):
        self.stem.eval()
        self.branch.eval()
        
    def train_both_stem_branch(self):
        self.stem.train()
        self.branch.train()

    def eval_both_stem_branch(self):
        self.stem.eval()
        self.branch.eval()
    
    def freeze_stem_train_branch(self):
        for name, param in self.stem.named_parameters():
            param.requires_grad = False
            
        for name, param in self.branch.named_parameters():
            param.requires_grad = True
    
    def inherit_attributes(self):
        assert self.stem.planes_per_block == self.branch.planes_per_block
        assert self.stem.diverged_planes_per_block == self.branch.diverged_planes_per_block

        assert self.stem.block_fn == self.branch.block_fn

        assert self.stem.divergence_idx == self.branch.divergence_idx
        assert self.stem.nb_branch == self.branch.nb_branch
        assert self.stem.nb_classes == self.branch.nb_classes
        assert self.stem.shift == self.branch.shift
        
        self.planes_per_block = self.stem.planes_per_block
        self.diverged_planes_per_block = self.stem.diverged_planes_per_block
        self.block_fn = self.stem.block_fn
        self.divergence_idx = self.stem.divergence_idx
        self.nb_branch = self.stem.nb_branch
        self.nb_classes = self.stem.nb_classes
        self.shift = self.stem.shift


class SubResNet152(nn.Module):
    def __init__(self, model, start_idx=None, end_idx=None, 
                 nb_blocks=[3, 8, 36, 3]):
        """
        nb_blocks should be passed to SubResNet152, not to ResNet152Helper 
        
        Passing nb_blocks to ResNet152Helper has been disabled since that is only 
        useful to create shallow monolithic model (less depth without branches). 
        
        model must be a complete ResNet152
        
        start_idx: included
        end_idx: not included
        
        For example, 
            if one wants to keep the whole layer3, 
            the whole layer4, and all the following layers 
            (global_pool and fc), then 
            start_idx=15, end_idx=56.
            
            if one wants to keep all early layers and stop before 
            the first block of layer3, then
            start_idx=0, end_idx=16.
        """
        super().__init__()
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.nb_blocks = nb_blocks
        
        self.inherit_attributes(model)
        self.extract_blocks(model)
        
        

    def extract_blocks(self, model):
        """
        For ResNet152:
        3 + 8 + 36 + 3 = 50
        50 + 4 = 54 (conv1, bn1, act1, maxpool)
        54 + 2 = 56 (global_pool, fc)
        """

        all_blocks = []

        # conv1 part
        all_blocks.append(model.conv1)
        all_blocks.append(model.bn1)
        all_blocks.append(model.act1)
        all_blocks.append(model.maxpool)
        for name, block in model.layer1.named_children():
            all_blocks.append(block)
        for name, block in model.layer2.named_children():
            all_blocks.append(block)
        for name, block in model.layer3.named_children():
            all_blocks.append(block)
        for name, block in model.layer4.named_children():
            all_blocks.append(block)
        all_blocks.append(model.global_pool)
        all_blocks.append(model.fc)

        if self.start_idx is None or self.end_idx is None:
            # this will keep all the 56 "modules"
            self.start_idx = 0
            self.end_idx = 56
        self.blocks = nn.ModuleList()

        for i in range(self.start_idx, self.end_idx):
            if self.skip_this_block(i):
                continue
            self.blocks.append(all_blocks[i])
    
    
    def skip_this_block(self, idx):
        """
        Use self.nb_blocks to check whether 
        idx falls within the range of 
        selected blocks. 
        
        self.start_idx: included
        self.end_idx: not included
        """
        if idx >= 0 and idx <= 3:
            return False
        if idx >= 54:
            return False
        
        if idx >= 4 and idx <= 6:
            # layer1 part (3 blocks)
            phase_idx = 0
            first_block_idx = 4
            nxt_phase_first_block_idx = 7
        elif idx >= 7 and idx <= 14:
            # layer2 part (8 blocks)
            phase_idx = 1
            first_block_idx = 7
            nxt_phase_first_block_idx = 15
        elif idx >= 15 and idx <= 50:
            # layer3 part (36 blocks)
            phase_idx = 2
            first_block_idx = 15
            nxt_phase_first_block_idx = 51
        elif idx >= 51 and idx <= 53:
            # layer4 part (3 blocks)
            phase_idx = 3
            first_block_idx = 51
            nxt_phase_first_block_idx = 54
        
        self.sanity_check_compatibility_nb_blocks_divergence_idx(first_block_idx, 
                                                                 nxt_phase_first_block_idx, 
                                                                 phase_idx)
        
        if self.start_idx == self.divergence_idx and \
            self.divergence_idx - first_block_idx >= self.nb_blocks[phase_idx] and \
                idx < nxt_phase_first_block_idx:
            # useful for splitting branches
            return True
        elif idx - first_block_idx >= self.nb_blocks[phase_idx]:
            # idx - first_block_idx:
            #   number of blocks that are already added in the current phase
            return True
        else:
            return False
    
    def sanity_check_compatibility_nb_blocks_divergence_idx(self, 
                                                            first_block_idx, 
                                                            nxt_phase_first_block_idx, 
                                                            phase_idx):
        # The block with global_idx=self.divergence_idx is the first block of
        # the branch, it is the first thin block and it is also
        # responsible for adapting the number of channels.

        # (self.divergence_idx - first_block_idx) is the number of blocks of
        # the current phase which will be put into the stem, these blocks are all
        # original blocks (not splitted).

        # If self.nb_blocks[phase_idx] <= self.divergence_idx - first_block_idx,
        # then all non-skipped blocks will be in the stem and the branch will
        # not have blocks in this phase. The first block of the branch will be
        # the first block of the next phase, which expects a thin version of
        # number of input channels. However the channel-adapter block was skipped,
        # it was skipped by the stem because there were no more quota in
        # self.nb_blocks[phase_idx].

        # if self.nb_blocks is not determined in advance, then a safe choice is to
        # set self.divergence_idx == self.divergence_idx [4 or 7 or 15 or 51].
        # The advantage of setting self.divergence_idx == self.divergence_idx is to
        # ensure that the channel-adapter block will always be put into the stem, so
        # the first block of the next phase (put into the branch) can receive a
        # correct input shape.
        if self.divergence_idx is not None and \
            self.divergence_idx >= first_block_idx \
                and self.divergence_idx < nxt_phase_first_block_idx:
            assert self.nb_blocks[phase_idx] > self.divergence_idx - first_block_idx, \
                """nb_blocks configuration incompatible with divergence_idx, forward pass 
                will throw RuntimeError: number of channels not matched."""
        
    def inherit_attributes(self, model):
        self.planes_per_block = model.planes_per_block
        self.diverged_planes_per_block = model.diverged_planes_per_block

        self.block_fn = model.block_fn

        self.divergence_idx = model.divergence_idx
        self.nb_branch = model.nb_branch
        self.nb_classes = model.nb_classes
        self.shift = model.shift
            
    
    def forward(self, x):
        for module in self.blocks:
            x = module(x)
        return x
    
    def __getitem__(self, idx):
        return self.blocks[idx]
    
    def __len__(self):
        return len(self.blocks)


class ResNet152Helper(nn.Module):
    block_kwargs = {"reduce_first": 1, "dilation": 1, "drop_block": None, "cardinality": 1,
                    "base_width": 64, "act_layer": nn.ReLU, "norm_layer": nn.BatchNorm2d,
                    "aa_layer": None}
    down_kwargs = {'kernel_size': 1,
                   'dilation': 1, 'first_dilation': 1, 'norm_layer': nn.BatchNorm2d}

    def __init__(self, nb_classes=10, divergence_idx=None, nb_branch=None, shift=0):
        """
        divergence_idx: 
            int (included) or None (no divergence, no split)
            diverge from which block (w.r.t. self.overall_block_idx)
            
            divergence_idx == 4 means diverging from the first block of layer1
            divergence_idx == 7 means diverging from the first block of layer2
            divergence_idx == 15 means diverging from the first block of layer3
            divergence_idx == 51 means diverging from the first block of layer4
            if divergence_idx not in [4, 7, 15, 51] and divergence_idx >= 5 and 
                divergence_idx <= 53, then a new downsample block (1 conv + 1 bn) 
                will be created, this new downsample block does not have pretrained 
                weights. Be careful. 
        nb_branch: 
            int or None (no divergence, no split into branches)
            how many branches do we need?
            
        self.overall_block_idx:
            # conv1 part
            0: conv1
            1: bn1
            2: act1
            3: maxpool
            
            # layer1 part (3 blocks)
            4, 5, 6
            
            # layer2 part (8 blocks)
            7, 8, 9, 10, 11, 12, 13, 14
            
            # layer3 part (36 blocks)
            15, 16, 17, 18, ..., 48, 49, 50
            
            # layer4 part (3 blocks)
            51, 52, 53
            
            global_pool: 54
            fc: 55
        """
        super().__init__()

        self.block_fn = Bottleneck
        self.inplanes = 64
        self.overall_block_idx = 0
        self.divergence_idx = divergence_idx
        self.nb_branch = nb_branch
        self.nb_classes = nb_classes
        self.shift = shift
        
        if self.divergence_idx is not None:
            # diverge in blocks from layer1, layer2, layer3, layer4
            assert self.divergence_idx >= 4
            assert self.divergence_idx <= 53

        self.nb_blocks = [3, 8, 36, 3]
        self.planes_per_block = [64, 128, 256, 512]
        self.diverged_planes_per_block = self.compute_diverged_planes_per_block()
        
        assert self.diverged_planes_per_block[0] >= 1
        assert self.diverged_planes_per_block[1] >= 1
        assert self.diverged_planes_per_block[2] >= 1
        assert self.diverged_planes_per_block[3] >= 1
        
        # conv1 part
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1,
                                  affine=True, track_running_stats=True)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.overall_block_idx += 4 # conv1, bn1, act1, maxpool
        
        self.layer1 = self.make_stage(0, stride=1)
        self.layer2 = self.make_stage(1, stride=2)
        self.layer3 = self.make_stage(2, stride=2)
        self.layer4 = self.make_stage(3, stride=2)

        self.global_pool = SelectAdaptivePool2d(pool_type="avg", flatten=True)
        self.fc = nn.Linear(self.diverged_planes_per_block[-1] * self.block_fn.expansion,
                            self.nb_classes, bias=True)
        self.overall_block_idx += 2 # global_pool, fc
        
    
    def make_stage(self, stage_idx, stride):
        planes = self.planes_per_block[stage_idx]
        blocks = []
        for block_idx in range(self.nb_blocks[stage_idx]):
            
            if self.divergence_idx is not None and self.nb_branch > 1:
                if self.overall_block_idx >= self.divergence_idx:
                    planes = self.diverged_planes_per_block[stage_idx]

            if block_idx == 0:
                # downsampling with stride=2 for the first block
                stride = stride  # 2
                downsample = downsample_conv(
                    in_channels=self.inplanes,
                    out_channels=planes * self.block_fn.expansion,
                    stride=stride, **self.down_kwargs)
            elif self.divergence_idx is not None \
                and self.nb_branch > 1 \
                    and self.overall_block_idx == self.divergence_idx:
                # change channel dimensionality without downsampling 
                stride = 1  
                downsample = downsample_conv(
                    in_channels=self.inplanes,
                    out_channels=planes * self.block_fn.expansion,
                    stride=stride, **self.down_kwargs)
            else:
                stride = 1
                downsample = None

            blocks.append(self.block_fn(
                self.inplanes, planes, stride, downsample, first_dilation=1,
                drop_path=None, **self.block_kwargs))

            self.inplanes = planes * self.block_fn.expansion
            self.overall_block_idx += 1

        stage = nn.Sequential(*blocks)

        return stage

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)
        x = self.fc(x)

        return x

    
    def compute_diverged_planes_per_block(self):
        def t(x, factor, shift):
            """transform helper function"""
            return int(np.floor(x / factor)) - shift
        
        # self.planes_per_block = [64, 128, 256, 512]
        
        if self.nb_branch is None or self.nb_branch == 1:
            diverged_planes_per_block = self.planes_per_block
        elif not isinstance(self.nb_branch, int) or self.nb_branch < 1:
            raise Exception("nb_branch={} not valid.".format(self.nb_branch))
        else:
            factor = np.sqrt(self.nb_branch)
            shift = self.shift
            if self.divergence_idx >= 4 and self.divergence_idx <= 6:
                # layer1
                diverged_planes_per_block = [t(self.planes_per_block[0], 
                                               factor, shift),
                                             t(self.planes_per_block[1],
                                               factor, shift),
                                             t(self.planes_per_block[2],
                                               factor, shift),
                                             t(self.planes_per_block[3], 
                                               factor, shift)]
            elif self.divergence_idx >= 7 and self.divergence_idx <= 14:
                # layer2
                diverged_planes_per_block = [self.planes_per_block[0],
                                             t(self.planes_per_block[1],
                                               factor, shift),
                                             t(self.planes_per_block[2],
                                               factor, shift),
                                             t(self.planes_per_block[3], 
                                               factor, shift)]
            elif self.divergence_idx >= 15 and self.divergence_idx <= 50:
                # layer3
                diverged_planes_per_block = [self.planes_per_block[0],
                                             self.planes_per_block[1],
                                             t(self.planes_per_block[2],
                                               factor, shift),
                                             t(self.planes_per_block[3], 
                                               factor, shift)]
            elif self.divergence_idx >= 51 and self.divergence_idx <= 53:
                # layer4
                diverged_planes_per_block = [self.planes_per_block[0],
                                             self.planes_per_block[1],
                                             self.planes_per_block[2],
                                             t(self.planes_per_block[3], 
                                               factor, shift)]
            else:
                raise Exception(
                    "divergence_idx={} is not valid.".format(self.divergence_idx))

        return diverged_planes_per_block
    

def calibrate_shift(divergence_idx, nb_branch, nb_classes, shift_limit=100, 
                    verbose=False, return_details=False, nb_blocks=None):
    """
    for nb_branch in range(1, 11):
        shift = calibrate_shift(divergence_idx=15,
                                nb_branch=nb_branch, nb_classes=62, verbose=True)
    """
    
    for shift in range(shift_limit):
        model = SubResNet152(ResNet152Helper(
            nb_classes=nb_classes, 
            divergence_idx=None, 
            nb_branch=None, shift=0), start_idx=divergence_idx, end_idx=56, 
                             nb_blocks=nb_blocks)

        a = count_trainable_parameters(model)
        
        model2 = SubResNet152(ResNet152Helper(
            nb_classes=nb_classes, 
            divergence_idx=divergence_idx, 
            nb_branch=nb_branch, shift=shift), start_idx=divergence_idx, end_idx=56,
                              nb_blocks=nb_blocks)
        
        diverged_planes_per_block = model2.diverged_planes_per_block
        
        b = count_trainable_parameters(model2) * nb_branch
        
        percent = b / a
        
        if percent <= 1:
            if verbose:
                print(model.diverged_planes_per_block)
                print(model2.diverged_planes_per_block)
                print(percent)
                print(nb_branch, shift)
                print()
            if return_details:
                return shift, diverged_planes_per_block, percent
            else:
                return shift
    raise Exception("shift_limit={} cannot get a satisfatory shit.".format(shift_limit))


def verify_id_and_total_size(divergence_idx, nb_branch, nb_classes, 
                             split_pretrained_weights, verbose=False):
    """
    Unit test
    
    First verifies whether the stem is actually shared and that the 
    branches are not shared, then verifies that the total size of the 
    diverged model does not surpass that of the monolithic model. 
    
    How to use this function:
    
    verify_id_and_total_size(divergence_idx=15, nb_branch=2, 
                             nb_classes=62, 
                             split_pretrained_weights=True, 
                             verbose=True)
    """
    import time
    
    t0 = time.time()
    
    device = get_device(verbose=True)
    
    pretrained_model = timm.create_model("resnet152",
                                         pretrained=True,
                                         num_classes=nb_classes)
    pretrained_model.to(device)

    models = get_diverged_resnet152_models(
        divergence_idx=divergence_idx, nb_branch=nb_branch,
        nb_classes=nb_classes, 
        split_pretrained_weights=split_pretrained_weights,
        device=device, concat=True)
    
    
    stem_ids, branch_ids = [], []
    for model in models:
        stem_ids.append(id(model.stem))
        branch_ids.append(id(model.branch))
    
    # verify that stem is shared, branches are not shared
    assert pd.Series(stem_ids).nunique() == 1
    assert pd.Series(branch_ids).nunique() == nb_branch
    
    if verbose:
        print("Good, stem is shared, branches are not shared.")
    
    # the following verifies the total size
    stem = models[0].stem
    branches = []
    for model in models:
        branches.append(model.branch)
    

    a = count_trainable_parameters(pretrained_model)
    b = count_trainable_parameters(stem)

    c = []
    for i, branch in enumerate(branches):
        c.append(count_trainable_parameters(branch))

    if verbose:
        print("pretrained_model =", a)
        print("stem =", b)
        print("branches =", c)
        print("stem + sum(branches) =", b + sum(c))
        print("pretrained_model - (stem + sum(branches)) =", a - (b + sum(c)))
        print("(stem + sum(branches)) / pretrained_model = {:.2f}%".format(
            (b + sum(c)) / a * 100))

    assert (b + sum(c)) / a <= 1
    
    print("verify_id_and_total_size done in {:.1f} s.".format(time.time() - t0))


def get_keys_before_divergence(model, divergence_idx):
    """
    harded-coded divergence_idx mapping
    
    used to select keys from pretrained_model.state_dict().keys() 
    for loading the stem part (model) of the sub-model
    """
    # divergence_idx should be after the layer2 (conv3_x) part of ResNet152
    assert divergence_idx >= 15
    
    key_prefix = ["conv1", "bn1", "layer1", "layer2"]
    
    if divergence_idx >= 15 and divergence_idx <= 50:
        for idx in range(divergence_idx - 15):
            key_prefix.append("layer3.{}.".format(idx))
    
    if divergence_idx >= 51 and divergence_idx <= 53:
        key_prefix.append("layer3")
        for idx in range(divergence_idx - 51):
            key_prefix.append("layer4.{}.".format(idx))
            
    if divergence_idx in [54, 55]:
        key_prefix.append("layer3")
        key_prefix.append("layer4")
        
    if divergence_idx > 55:
        key_prefix.append("layer3")
        key_prefix.append("layer4")
        key_prefix.append("fc")
        
    
    key_prefix = tuple(key_prefix)
    
    
    model_dict = model.state_dict()
    kept_keys = []
    for k, v in model_dict.items():
        if k.startswith(key_prefix):
            kept_keys.append(k)
    return kept_keys


def load_pretrained_to_stem(pretrained_model, stem, divergence_idx, verbose=False):
    kept_keys = get_keys_before_divergence(pretrained_model, divergence_idx)

    stem_state_dict = {k: pretrained_model.state_dict()[k] for k in kept_keys}

    missing_keys, unexpected_keys = stem.load_state_dict(
        stem_state_dict, strict=False)

    if verbose:
        print(len(missing_keys))
        print()
        print(len(unexpected_keys))
    return stem


def get_stem(divergence_idx, nb_branch, nb_classes, shift, 
             device, pretrained_model, nb_blocks):
    stem = ResNet152Helper(
        nb_classes=nb_classes, divergence_idx=divergence_idx, nb_branch=nb_branch, shift=shift)

    stem = load_pretrained_to_stem(pretrained_model, stem,
                                   divergence_idx, verbose=False)
    
    stem = SubResNet152(stem, start_idx=0,
                        end_idx=divergence_idx, nb_blocks=nb_blocks)
    
    stem.to(device)
    
    return stem


def is_bn(k):
    """
    k: str, parameter name,
    """
    if k.startswith("bn"):
        return True
    if "running_mean" in k:
        return True
    if "running_var" in k:
        return True
    if "num_batches_tracked" in k:
        return True
    if ".bn" in k:
        return True
    if "downsample.1" in k:
        return True
    return False

def is_conv(k):
    """
    k: str, parameter name,
    """
    if "conv" in k:
        return True
    if "downsample.0" in k:
        return True
    return False


def is_fc(k):
    """
    k: str, parameter name,
    """
    if "fc" in k:
        return True
    return False


def generate_pruned_weights_per_layer(k, pretrained_dict, branch_dict, nb_branch):
    """
    k: str, key in pretrained_dict or branch_dict
    
    returns new_branch_dict_list
        A list of torch.Tensor, the size of the list is nb_branch.
        It is the pruned pretrained weights (state_dict) for the 
        layer named by k for each of the nb_branch branches. 
        
    input_channel_ratio = branch_dict[k].shape[1] / pretrained_dict[k].shape[1]
    output_channel_ratio = branch_dict[k].shape[0] / pretrained_dict[k].shape[0]
    
    output_channel_ratio < 1 for all k
    input_channel_ratio < 1 for almost all k except for the very first conv layer 
    and the very first downsample conv layer (for example, in the case of 
    divergence_idx=15, layer3.0.conv1.weight, layer3.0.downsample.0.weight). For 
    these 2 conv layers, input_channel_ratio == 1. 
    """
    pruned_tensors = []
    with torch.no_grad():
        for i in range(nb_branch):
            new_tensor = torch.zeros_like(branch_dict[k])
            pruned_tensors.append(new_tensor)
            
        for in_channel in range(branch_dict[k].shape[1]):
            pretrained_output_channels = pretrained_dict[k].shape[0]
            new_output_channels = branch_dict[k].shape[0]
            
            # compute output_channel_indices
            if pretrained_output_channels >= new_output_channels * nb_branch:
                output_channel_indices = np.asarray(
                    list(range(new_output_channels * nb_branch)))
            else:
                output_channel_indices = list(range(pretrained_output_channels))
                
                while len(output_channel_indices) < new_output_channels * nb_branch:
                
                    output_channel_indices.extend(np.random.choice(
                        pretrained_output_channels, 
                        size=pretrained_output_channels,
                        replace=False).tolist())
                
                output_channel_indices = output_channel_indices[:new_output_channels * nb_branch]
                output_channel_indices = np.asarray(output_channel_indices)
            
            # copy weights per input channel
            start_ = 0
            end_ = pruned_tensors[0].shape[0]
            for i in range(nb_branch):
                pruned_tensors[i][:, in_channel, :, :] = \
                    pretrained_dict[k][output_channel_indices[start_:end_], \
                        in_channel, :, :]
                start_ += pruned_tensors[0].shape[0]
                end_ += pruned_tensors[0].shape[0]
    
    return pruned_tensors


def load_pretrained_to_branches(pretrained_model, branches, divergence_idx, verbose=False):
    """
    keys_after_divergence can be partitioned into 3 parts:
        conv keys (after divergence): 1 key per Conv2d layer if bias=False
        bn keys (after divergence): 5 keys per BatchNorm2d layer
        fc keys (after divergence): fc.weight, fc.bias
        
    fc layer will never need to be copied
    
    bn layers will never be copied either, they will be trained from scratch using new data
    
    conv layers can be either random initialization or copied from part of the whole model
        (sampling of 2d filters with replacement or wihout replacement)
    
    newly-added conv layers in the skip-connections will be trained from scratch
    
    conv_keys_after_divergence is a subset of pretrained_model.state_dict().keys(), this 
    means that newly-added conv layers in the skip-connections will not show up in 
    conv_keys_after_divergence. 
    
    Only layers in conv_keys_after_divergence will get weights from the pretrained model.
    """
    stem_keys = get_keys_before_divergence(pretrained_model, divergence_idx)

    pretrained_dict = pretrained_model.state_dict()
    conv_keys_after_divergence = [k for k in pretrained_dict.keys()
        if (k not in stem_keys) and is_conv(k)]
    
    nb_branch = len(branches)
    branch = branches[0]
    
    # collections.OrderedDict: from <class 'str'> to <class 'torch.Tensor'>
    branch_dict = branch.state_dict()

    new_branch_dict_list = []
    for i in range(nb_branch):
        new_branch_dict_list.append(OrderedDict())
    for k in conv_keys_after_divergence:
        pruned_tensors = generate_pruned_weights_per_layer(k,
            pretrained_dict, branch_dict, nb_branch)
        for i in range(nb_branch):
            new_branch_dict_list[i][k] = pruned_tensors[i]
    
    for i, branch in enumerate(branches):
        missing_keys, unexpected_keys = branch.load_state_dict(
            new_branch_dict_list[i], strict=False)

        if verbose:
            print("Branch {}".format(i))
            print(len(missing_keys))
            print()
            print(len(unexpected_keys))
            print()
    return branches


def get_branches(divergence_idx, nb_branch, nb_classes, 
                 shift, device, pretrained_model=None, only_first=False, 
                 nb_blocks=None):
    
    branches = []
    
    nb_branch_create = nb_branch
    if only_first:
        nb_branch_create = 1
    
    for i in range(nb_branch_create):
        branch = ResNet152Helper(nb_classes=nb_classes, 
                                 divergence_idx=divergence_idx, 
                                 nb_branch=nb_branch, 
                                 shift=shift)
        branches.append(branch)
        
    if pretrained_model is not None:
        branches = load_pretrained_to_branches(pretrained_model, branches, 
                                               divergence_idx, verbose=False)
    
    pruned_branches = []
    for i in range(nb_branch_create):
        branch = SubResNet152(branches[i], start_idx=divergence_idx, 
                              end_idx=56, nb_blocks=nb_blocks)
        branch.to(device)
        pruned_branches.append(branch)
    return pruned_branches


def list2str(inp):
    """
    Example input: [3, 8, 36, 3]
    returns "3_8_36_3"
    """
    return "_".join([str(xx) for xx in inp])


def lookup_precomputed_shift(divergence_idx, nb_branch, nb_classes, 
                             nb_blocks=[3, 8, 36, 3]):
    
    shift_path = "precomputed_shift_lookup_table_{}.csv"
    
    shift_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), shift_path.format(list2str(nb_blocks)))
    
    assert os.path.exists(
        shift_path), "{} not found, please run precompute_shift.py to get it.".format(shift_path)
    
    shift_df = pd.read_csv(shift_path)

    row_mask = (shift_df["divergence_idx"] == divergence_idx) \
        & (shift_df["nb_branch"] == nb_branch)\
            & (shift_df["nb_classes"] == nb_classes)
    shift = shift_df.loc[row_mask, "shift"].item()
    
    # - 99999 is a marker for failed configuration, see precompute_shift.py
    assert shift != - 99999, \
        "divergence_idx={}, nb_branch={}, nb_classes={} cannot be instantiated.".format(
            divergence_idx, nb_branch, nb_classes)
    
    return shift


def warm_stem_BatchNorm(stem, train_dataloader, device, warm_stem_epochs):
    """
    warm stem's batch norm layers with several 
    forward passes of training data, then set 
    stem to eval mode
    """
    stem.train()
    with torch.no_grad():
        for i in range(warm_stem_epochs):
            for X, y in train_dataloader:
                X, y = X.to(device), y.to(device)
                logits = stem(X)
    stem.eval()
    return stem


def get_diverged_resnet152_models(divergence_idx, nb_branch, nb_classes, 
                                  split_pretrained_weights, device="cpu",
                                  train_dataloader=None, warm_stem_epochs=0, 
                                  concat=True, only_first=False, 
                                  pretrained_model="resnet152",
                                  nb_blocks=[3, 8, 36, 3]):
    """
    model.load_state_dict(pretrained_model.state_dict()) returns:
        missing_keys: 
            keys which are present in model but not in pretrained_model
        unexpected_keys:
            keys which are present in pretrained_model but not in model
            
    If warm_stem_epochs > 0, then warm stem's batch norm 
    layers with several forward passes of training data, then set 
    stem to eval mode. train_dataloader cannot be None.
    
    If concat=False, returns a stem model and a list of branch models.
    If concat=True, returns a list of concatenated models, where 
    each model concatenates a shared stem model and its own branch model. 
    
    If split_pretrained_weights=True, then split the pretrained model's 
    weight tensors to initialize the branches.
    If split_pretrained_weights=False, then the branches will use random 
    initialization. 
    """
    if warm_stem_epochs > 0:
        assert train_dataloader is not None
    
    pretrained_model = timm.create_model(pretrained_model,
                                         pretrained=True,
                                         num_classes=nb_classes)
    
    try:
        shift = lookup_precomputed_shift(divergence_idx, 
                                        nb_branch, nb_classes, 
                                        nb_blocks)
    except Exception as e:
        warnings.warn("Cannot lookup precomputed shift table: {}".format(e))
        shift = 0
    
    stem = get_stem(divergence_idx, nb_branch, 
                    nb_classes, shift, device, pretrained_model, 
                    nb_blocks=nb_blocks)
    
    if split_pretrained_weights:
        branches = get_branches(divergence_idx, nb_branch,
                                nb_classes, shift, device, 
                                pretrained_model=pretrained_model, 
                                only_first=only_first,
                                nb_blocks=nb_blocks)
    else:
        branches = get_branches(divergence_idx, nb_branch,
                                nb_classes, shift, device, 
                                pretrained_model=None, 
                                only_first=only_first,
                                nb_blocks=nb_blocks)
    
    if warm_stem_epochs > 0:
        stem = warm_stem_BatchNorm(
            stem, train_dataloader, device, warm_stem_epochs)

    if not concat:
        return stem, branches
    else:
        models = []
        for branch in branches:
            model = ConcatenatedModel(stem, branch)
            models.append(model)
        return models
    

def test_with_data(divergence_idx, nb_branch, nb_classes, split_pretrained_weights):
    """
    Unit test
    """
    import time
    import multiprocessing
    from torchvision import transforms
    from utils import Icdar2003CharacterDataset, Resnet152cfg
    
    img_size = 32
    batch_size = 32
    
    device = get_device(verbose=True)
    
    num_workers = multiprocessing.cpu_count()
    print("multiprocessing.cpu_count() = {}".format(num_workers))


    t0 = time.time()
    
    transform = transforms.Compose([lambda x: x.convert("RGB"),
                                    transforms.Resize([img_size, img_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=Resnet152cfg.mean,
                                                         std=Resnet152cfg.std)])
    
    train_dataset = Icdar2003CharacterDataset(train_split=True,
                                              enforce_all_alphanumeric_classes=True,
                                              transform=transform,
                                              dataset_root=os.path.join(os.pardir, 
                                                  "datasets",
                                                  "ICDAR_2003_character_dataset"))
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    print("Data ready in {:.2f} s.".format(time.time() - t0))
    t0 = time.time()
    
    models = get_diverged_resnet152_models(divergence_idx=divergence_idx, 
                                           nb_branch=nb_branch,
                                           nb_classes=nb_classes,
                                           split_pretrained_weights=split_pretrained_weights,
                                           device=device, concat=True, 
                                           train_dataloader=train_dataloader, 
                                           warm_stem_epochs=10)
    
    print("get_diverged_resnet152_models ready in {:.2f} s.".format(
        time.time() - t0))
    t0 = time.time()
    
    
    
    for model in models:
        model.train()
        model.freeze_stem_train_branch()
    
    for batch_idx, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        for model in models:
            logits = model(X)
        
        if batch_idx >= 1:
            break
    
    print("Training phase ready in {:.2f} s.".format(
        time.time() - t0))
    
    return


if __name__ == "__main__":
    
    divergence_idx = 15
    nb_branch = 3
    nb_classes = 62
    split_pretrained_weights = True
    
    verify_id_and_total_size(divergence_idx=divergence_idx, nb_branch=nb_branch,
                             nb_classes=nb_classes,
                             split_pretrained_weights=split_pretrained_weights,
                             verbose=True)
    
    test_with_data(divergence_idx=divergence_idx, 
                   nb_branch=nb_branch, 
                   nb_classes=nb_classes, 
                   split_pretrained_weights=split_pretrained_weights)
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
