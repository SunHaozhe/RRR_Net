
import os
import math
import random
import json
from collections import defaultdict
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
import wandb

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


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

def set_random_seeds(random_seed):
    if random_seed is not None:
        torch.backends.cudnn.deterministic = True
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)


def get_torch_gpu_environment():
    env_info = dict()
    env_info["PyTorch_version"] = torch.__version__

    if torch.cuda.is_available():
        env_info["cuda_version"] = torch.version.cuda
        env_info["cuDNN_version"] = torch.backends.cudnn.version()
        env_info["nb_available_GPUs"] = torch.cuda.device_count()
        env_info["current_GPU_name"] = torch.cuda.get_device_name(
            torch.cuda.current_device())
    else:
        env_info["nb_available_GPUs"] = 0
    return env_info


def get_library_version():
    import sys
    
    library_check_list = ["torch", "timm"]
    
    library_version = dict()
    for library_name in library_check_list:
        if library_name in sys.modules:
            library_version["{}_version".format(
                library_name)] = globals()[library_name].__version__
    return library_version


def report_trainable_params_count(model, use_wandb):
    trainable_params = count_trainable_parameters(model)
    print("count_trainable_parameters(model) = {}".format(trainable_params))
    if use_wandb:
        wandb.run.summary["count_trainable_parameters"] = trainable_params


def load_weights_to_unmatched_model(new_model, pretrained_model_state_dict, 
                                    verbose=False, use_wandb=False):
    """
    model.load_state_dict(pretrained_model.state_dict()) returns:
        missing_keys: 
            keys which are present in model but not in pretrained_model
        unexpected_keys:
            keys which are present in pretrained_model but not in model
            
    If new_model is a superset of pretrained_model, 
    missing_keys should not be empty 
    (len(missing_keys)=15 per missing blocks for ResNet152), 
    unexpected_keys should be empty. 
    """
    missing_keys, unexpected_keys = new_model.load_state_dict(
        pretrained_model_state_dict, strict=False)

    if verbose:
        print("len(missing_keys) = {}".format(len(missing_keys)))
        print("len(unexpected_keys) = {}".format(len(unexpected_keys)))
    if use_wandb:
        wandb.run.summary["len_missing_keys"] = len(missing_keys)
        wandb.run.summary["len_unexpected_keys"] = len(unexpected_keys)
    return new_model


def get_optimizer(args, model):
    if args.optimizer == "AdamW":
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
    return optim


def get_lr_scheduler(args, optim, train_dataloader):
    if args.lr_scheduler == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optim, max_lr=args.OneCycleLR_max_lr, 
            steps_per_epoch=len(train_dataloader), 
            epochs=args.epochs)
    elif args.lr_scheduler == "CosineAnnealingWarmRestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optim, 
            T_0=args.CosineAnnealingWarmRestarts_T_0, 
            T_mult=args.CosineAnnealingWarmRestarts_T_mult)
    else:
        scheduler = None
    return optim, scheduler


def compute_class_support(train_dataset, nb_classes):
    class_support = [0] * nb_classes
    classes = np.asarray(train_dataset.classes_)
    for k in range(nb_classes):
        class_support[k] += (classes == k).sum().item()
    return class_support


def create_outlying_label(target, class_support, nb_classes, alpha=1):
    """
    https://arxiv.org/pdf/2107.02393.pdf
    Section 3.3, Algorithm 1
    
    Cifar-10 (classification): alpha=5
    Cifar-100 (classification): alpha=1
    Long-tailed Food101 (classification): alpha=1
    CamVid (semantic segmentation): alpha=6
    
    target: 
        <class 'torch.Tensor'> 
        torch.Size([batch_size]) 
        torch.int64
        
    new_target: 
        <class 'torch.Tensor'> 
        torch.Size([batch_size, nb_classes]) 
        torch.float32
    
    examples_curr_class:
        <class 'torch.Tensor'> 
        torch.Size([batch_size]) 
        torch.bool
    """
    with torch.no_grad():
        class_indices = np.argsort(class_support)
        new_target = F.one_hot(
            target, num_classes=nb_classes).to(torch.float32)
        N = 0
        for k in class_indices:
            examples_curr_class = (target == k)
            new_target[examples_curr_class, :] *= (N + 1)
            N += 1
        new_target *= alpha
    return new_target


def compute_loss(args, logits, y, loss_kwargs):
    # class_weights, num_classes
    if args.loss == "cross_entropy":
        loss = F.cross_entropy(logits, y)
    elif args.loss == "weighted_cross_entropy":
        loss = F.cross_entropy(logits, y, weight=loss_kwargs["class_weights"])
    elif args.loss == "MSE":
        y = F.one_hot(y, num_classes=loss_kwargs["nb_classes"]).to(torch.float32)
        loss = F.mse_loss(logits, y)
    elif args.loss == "MSE_outlying_1":
        y = create_outlying_label(y, loss_kwargs["class_support"], 
                                  loss_kwargs["nb_classes"], 
                                  alpha=1)
        loss = F.mse_loss(logits, y)
    elif args.loss == "MSE_outlying_5":
        y = create_outlying_label(y, loss_kwargs["class_support"], 
                                  loss_kwargs["nb_classes"], 
                                  alpha=5)
        loss = F.mse_loss(logits, y)
    elif args.loss == "focal_loss":
        # https://github.com/AdeelH/pytorch-multi-class-focal-loss 
        focal_loss = torch.hub.load(
            'adeelh/pytorch-multi-class-focal-loss',
            model='FocalLoss',
            alpha=None,
            gamma=2,
            reduction='mean',
            force_reload=False,
            verbose=False
        )
        loss = focal_loss(logits, y)
    elif args.loss == "weighted_focal_loss":
        focal_loss = torch.hub.load(
            'adeelh/pytorch-multi-class-focal-loss',
            model='FocalLoss',
            alpha=loss_kwargs["class_weights"],
            gamma=2,
            reduction='mean',
            force_reload=False,
            verbose=False
        )
        loss = focal_loss(logits, y)
    else:
        raise Exception("Uknown loss: {}".format(args.loss))
    return loss


class QuickAccuracyMetric:
    def __init__(self) -> None:
        self.reset()
    
    def reset(self):
        self.correct_cnt = 0
        self.total_cnt = 0

    def update(self, logits, target):
        self.correct_cnt += (logits.argmax(dim=1) == target).sum().item()
        self.total_cnt += target.shape[0]
    
    def compute(self, format="percentage"):
        res = self.correct_cnt / self.total_cnt
        if format == "percentage":
            res = res * 100
        return res


class AccuracyMetric:
    def __init__(self, nb_classes):
        self.nb_classes = nb_classes
        self.reset()
        
    def reset(self):
        self.correct_cnt = [0] * self.nb_classes
        self.total_cnt = [0] * self.nb_classes
        
    def update(self, logits, target):
        """
        logits: 
            <class 'torch.Tensor'> 
            torch.Size([batch_size, nb_classes]) 
            torch.float32
        target: 
            <class 'torch.Tensor'> 
            torch.Size([batch_size]) 
            torch.int64
            
        current_class: torch.Size([batch_size]) torch.bool
        correct_prediction: torch.Size([batch_size]) torch.bool
        """
        for k in range(self.nb_classes):
            current_class = (target == k)
            correct_prediction = (logits.argmax(dim=1) == target)
            self.correct_cnt[k] += (current_class * correct_prediction).sum().item()
            self.total_cnt[k] += current_class.sum().item()
            
    def compute(self, mode="all", format="percentage", ratio_worst_classes=0.3):
        """
        all: compute the accuracy by averaging all samples and classes.
        class_balanced: compute the accuracy for each class separately, 
            and average the accuracies across classes with equal weights 
            for each class.
        class_weighted: compute the accuracy for each class separately, 
            and average the accuracies across classes, weighting each 
            class by its support.
            
        format: percentage or anything else
        """
        if mode == "all":
            correct_cnt_ = 0
            total_cnt_ = 0
            for k in range(self.nb_classes):
                correct_cnt_ += self.correct_cnt[k]
                total_cnt_ += self.total_cnt[k]
            res = correct_cnt_ / total_cnt_
        elif mode == "class_balanced":
            """
            does not take into account classes that have 0 support
            """
            accuracy_per_class = []
            for k in range(self.nb_classes):
                if self.total_cnt[k] == 0:
                    continue
                else:
                    accuracy_per_class.append(
                        self.correct_cnt[k] / self.total_cnt[k])
            res = np.mean(accuracy_per_class)
        elif mode == "worst_classes":
            """
            takes into account classes that have 0 support
            """
            nb_worst_classes = max(1, math.floor(ratio_worst_classes * self.nb_classes))
            accuracy_per_class = []
            for k in range(self.nb_classes):
                if self.total_cnt[k] == 0:
                    score = 0
                else:
                    score = self.correct_cnt[k] / self.total_cnt[k]
                accuracy_per_class.append(score)
            accuracy_per_class = np.asarray(accuracy_per_class)
            class_indices = np.argsort(self.total_cnt)[:nb_worst_classes]
            res = np.mean(accuracy_per_class[class_indices])
        else:
            raise Exception("Unknown mode={}".format(mode))
        
        if format == "percentage":
            res = res * 100
        return res


class SimpleMLP(nn.Module):
    def __init__(self, img_size, img_mode, nb_classes):
        super().__init__()
        in_channels = 1
        if img_mode == "RGB":
            in_channels = 3
        input_size = img_size * img_size * in_channels
        hidden_layer_size = [128, 128]
        self.linear1 = nn.Linear(input_size, hidden_layer_size[0])
        self.linear2 = nn.Linear(hidden_layer_size[0], hidden_layer_size[1])
        self.linear3 = nn.Linear(hidden_layer_size[1], nb_classes)
        
        self.default_cfg = {"input_size": (in_channels, img_size, img_size),
                            "num_classes": nb_classes}
        if img_mode == "RGB":
            self.default_cfg["mean"] = (0.485, 0.456, 0.406)
            self.default_cfg["std"] = (0.229, 0.224, 0.225)
        else:
            self.default_cfg["mean"] = (0.449,)
            self.default_cfg["std"] = (0.226,)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self, img_mode, nb_classes):
        super().__init__()
        img_size = 32
        in_channels = 1
        if img_mode == "RGB":
            in_channels = 3
        
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(6, 2),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Linear(64, nb_classes)
        
        self.default_cfg = {"input_size": (in_channels, img_size, img_size),
                            "num_classes": nb_classes}
        if img_mode == "RGB":
            self.default_cfg["mean"] = (0.485, 0.456, 0.406)
            self.default_cfg["std"] = (0.229, 0.224, 0.225)
        else:
            self.default_cfg["mean"] = (0.449,)
            self.default_cfg["std"] = (0.226,)

    def forward(self, x):
        x = self.sequential(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



def get_model(model_name, device, pretrained, num_classes=0, 
              train_layers="only_last", args=None):
    if model_name.startswith("resnet"):
        
        assert args.img_mode == "RGB", "img_mode=L with resnet is not implemented"
        
        if train_layers == "random_init":
            pretrained = False
        
        model = timm.create_model(model_name,
                                pretrained=pretrained,
                                num_classes=num_classes)
        model.default_cfg["input_size"] = (3, args.img_size, args.img_size)

        if train_layers == "only_last":
            for name, param in model.named_parameters():
                if name.startswith(model.default_cfg["classifier"]):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif train_layers == "conv4_x":
            """
            conv4_x is the nomenclature of the main paper, 
            it is referred to as layer3 in timm
            """
            for name, param in model.named_parameters():
                if name.startswith(model.default_cfg["classifier"]):
                    param.requires_grad = True
                elif name.startswith(("layer3", "layer4")):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif train_layers in ["retrain_all", "random_init"]:
            for name, param in model.named_parameters():
                param.requires_grad = True
        else:
            raise NotImplementedError
    elif model_name.startswith("SimpleMLP"):
        nb_classes = 62
        model = SimpleMLP(args.img_size, args.img_mode, nb_classes)
    elif model_name.startswith("SimpleCNN"):
        nb_classes = 62
        model = SimpleCNN(args.img_mode, nb_classes)
    else:
        Exception("Uknown model_name: {}".format(model_name))

    model.to(device)
    return model


def get_transform(img_size, data_mean, data_std, args):
    proba = 1
    
    list_albumentations = [
        A.Affine(shear=10, p=proba),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.01,
                            rotate_limit=30, p=proba),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15,
                    b_shift_limit=15, p=proba),
        A.RandomBrightnessContrast(p=proba),
        A.Posterize(p=proba),
        A.MotionBlur(p=proba),
        A.ImageCompression(p=proba),
        A.GaussNoise(p=proba),
        A.Equalize(p=proba),
        A.Solarize(p=proba),
        A.Sharpen(p=proba),
        A.InvertImg(p=proba),
    ]
    
    transform = dict()
    
    transform["train"] = A.Compose(
        [
            A.Resize(height=img_size, width=img_size, interpolation=1),
            A.OneOf(list_albumentations, p=1),
            A.OneOf(list_albumentations, p=1),
            A.Normalize(mean=data_mean, std=data_std),
            ToTensorV2(),
        ]
    )
    transform["test"] = A.Compose(
        [
            A.Resize(height=img_size, width=img_size, interpolation=1),
            A.Normalize(mean=data_mean, std=data_std),
            ToTensorV2(),
        ]
    )

    return transform


def run_model_helper(model, X, model_name):
    """
    This function should only be used for model in train mode, 
    never use it for model in eval mode.
    
    This function is useful when input img_size <= 32, 
    however this function will not do the job when img_size <= 16.
    
    ResNet has 5 downsampling operation, an input image of size 
    32x32 (or smaller) will become 1x1 in the layer4 (conv5_x). If furthermore 
    the batch size is 1 (because of the last batch when drop_last=False 
    for DataLoader or other reasons), this will raise such as error: 
        ValueError: Expected more than 1 value per channel when 
        training, got input size torch.Size([1, channel_size, 1, 1]) 
    """
    # X has shape (batch_size, channel_size, img_size, img_size)
    if X.shape[0] == 1 and X.shape[-1] <= 32 and model_name.startswith("resnet"):
        if X.shape[-1] <= 16:
            raise NotImplementedError

        # convert the selected BatchNorm2d layers to eval mode
        for block_name, block in model.layer4.named_children():
            # the downsampling block:
            #   the downsampling in the main branch
            #       is done by layer4[0].conv2 for ResNet[50,101,152],
            #       so layer4[0].bn1 will not be converted to eval mode.
            #   the downsampling in the main branch
            #       is done by layer4[0].conv1 for ResNet[18,34],
            #       so BatchNorm layers will be converted to eval mode.
            if block_name == "0":
                ## skip-connection branch
                # model.layer4[0].downsample[1] is also a BatchNorm2d which
                # appears after a downsampling conv layer.
                model.layer4[0].downsample[1].eval()
                for layer_name, layer in model.layer4[0].named_children():
                    ## Main branch
                    if layer_name.startswith("bn"):
                        if model_name.startswith(("resnet152",
                                "resnet101", "resnet50")) and layer_name == "bn1":
                            continue
                        layer.eval()
            else:  # the following blocks
                for layer_name, layer in model.layer4[int(block_name)].named_children():
                    if layer_name.startswith("bn"):
                        layer.eval()

        logits = model(X)

        # convert the selected BatchNorm2d layers back to train mode
        model.train()
    else:
        logits = model(X)
    return logits


class Resnet152cfg:
    """
    Input statistics of timm's resnet152
    Used to construct data transforms
    
    Obtained from:
    model = timm.create_model("resnet152",
                              pretrained=True,
                              num_classes=62)
    print(model.default_cfg)
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    input_size = (3, 224, 224)
    num_classes = 1000
    interpolation = "bicubic"


def worker_init_fn(worker_id):
    """
    useful for PyTorch version < 1.9
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_multiprocessing_context(args):
    # https://github.com/pytorch/pytorch/issues/44687
    multiprocessing_context = None
    if hasattr(args, "nb_branch") and args.nb_branch is not None and args.nb_branch > 1 \
        and hasattr(args, "n_jobs") and args.n_jobs is not None and args.n_jobs > 1 \
            and hasattr(args, "num_workers") and args.num_workers is not None \
                and args.num_workers > 1:
        from joblib.externals.loky.backend.context import get_context
        multiprocessing_context = get_context('loky')
    return multiprocessing_context


def get_dataloaders(args):
    img_size = args.img_size
    data_mean = Resnet152cfg.mean
    data_std = Resnet152cfg.std

    transform = get_transform(img_size, data_mean, data_std, args)

    if args.dataset == "icdar03_char_micro":
        train_dataset, val_dataset, test_dataset = get_ICDAR_2003_character_datasets(transform, args)
    else:
        train_dataset, val_dataset, test_dataset = get_40_datasets(transform, args)

    multiprocessing_context = get_multiprocessing_context(args)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        multiprocessing_context=multiprocessing_context
    )
    
    if val_dataset is not None:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context
        )
    else:
        val_dataloader = None
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        multiprocessing_context=multiprocessing_context
    )
    
    return train_dataloader, val_dataloader, test_dataloader
    

s0_list = ["BCT_Micro", "BRD_Micro", "CRS_Micro", "FLW_Micro", "MD_MIX_Micro", 
           "PLK_Micro", "PLT_VIL_Micro", "RESISC_Micro", "SPT_Micro", "TEX_Micro"]

s1_list = ["ACT_40_Micro", "APL_Micro", "DOG_Micro", "INS_2_Micro", "MD_5_BIS_Micro",
           "MED_LF_Micro", "PLT_NET_Micro", "PNU_Micro", "RSICB_Micro", "TEX_DTD_Micro"]
    
s2_list = ["ACT_410_Micro", "AWA_Micro", "BTS_Micro", "FNG_Micro", "INS_Micro",
           "MD_6_Micro", "PLT_DOC_Micro", "PRT_Micro", "RSD_Micro", "TEX_ALOT_Micro"]

s3_list = ["ARC_Micro", "ASL_ALP_Micro", "BFY_Micro", "BRK_Micro", "MD_5_T_Micro", 
           "PLT_LVS_Micro", "POT_TUB_Micro", "SNK_Micro", "UCMLU_Micro", "VEG_FRU_Micro"]


def get_40_datasets(transform, args):

    dataset_name = args.dataset
    
    if dataset_name in s0_list:
        subfolder_name = "Set0_Micro"
    elif dataset_name in s1_list:
        subfolder_name = "Set1_Micro"
    elif dataset_name in s2_list:
        subfolder_name = "Set2_Micro"
    elif dataset_name in s3_list:
        subfolder_name = "Set3_Micro"
    else:
        raise Exception("Dataset {} not recognizable.".format(args.dataset))

    dataset_directory = os.path.join(
        os.pardir, "datasets", subfolder_name, dataset_name)

    train_dataset = Benchmark40MicroSplitDataset(
        dataset_directory=dataset_directory, transform=transform["train"], split="train", 
        train_examples_per_class=args.train_examples_per_class, 
        val_examples_per_class=args.val_examples_per_class,
        test_examples_per_class=args.test_examples_per_class)
    
    if args.val_examples_per_class > 0:
        val_dataset = Benchmark40MicroSplitDataset(
            dataset_directory=dataset_directory, transform=transform["test"], split="val", 
            train_examples_per_class=args.train_examples_per_class, 
            val_examples_per_class=args.val_examples_per_class,
            test_examples_per_class=args.test_examples_per_class)
    else:
        val_dataset = None
    
    test_dataset = Benchmark40MicroSplitDataset(
        dataset_directory=dataset_directory, transform=transform["test"], split="test", 
        train_examples_per_class=args.train_examples_per_class, 
        val_examples_per_class=args.val_examples_per_class,
        test_examples_per_class=args.test_examples_per_class)
    
    return train_dataset, val_dataset, test_dataset


def get_ICDAR_2003_character_datasets(transform, args):
    train_dataset = Icdar2003CharacterDataset(use_micro_version=True,
                                                micro_split="train",
                                                use_albumentations=True,
                                            transform=transform["train"],
                                            dataset_root=os.path.join(
                                                os.pardir, "datasets",
                                                "ICDAR_2003_character_dataset"))
    val_dataset = Icdar2003CharacterDataset(use_micro_version=True,
                                            micro_split="val",
                                            use_albumentations=True,
                                            transform=transform["test"],
                                            dataset_root=os.path.join(
                                                os.pardir, "datasets",
                                                "ICDAR_2003_character_dataset"))
    test_dataset = Icdar2003CharacterDataset(use_micro_version=True,
                                            micro_split="test",
                                                use_albumentations=True,
                                            transform=transform["test"],
                                            dataset_root=os.path.join(
                                                os.pardir, "datasets",
                                                "ICDAR_2003_character_dataset"))
    return train_dataset, val_dataset, test_dataset


class Icdar2003CharacterDataset(torch.utils.data.Dataset):
    """
    http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions
    
    ========================================================
    
    Folder structure: 
    
    ICDAR_2003_character_dataset
        TrialTrain_Set_6185_Characters
            char
                *.jpg
            char.xml
        TrialTest_Set_5430_Characters
            char
                *.jpg
            char.xml
            
    ========================================================
    
    training set has 75 classes in total, test set has 73 classes in 
        total. They have 71 classes in common (set intersection). 
    
    The 4 classes that are in the training set but not in the 
        test set (number in parentheses are the number of image 
        instances of that class): 'Ñ' (2), ':' (2), 'é' (1), 'É' (2), 
        which count for 7 images.
        
    The 2 classes that are in the test set but not in the training 
        set: ',' (6), 'z' (2), which count for 8 images.
        
    Let's recall that there are 6185 images in the training set, 5430 
        images in the test set.
        
    The 71 shared classes (71 = 26 + 25 + 10 + 10):
        It contains all the 26 uppercase basic Latin letters
        It contains 25 lowercase basic Latin letters except "z"
        It contains all the 10 ASCII digits (0, 1, 2, 3, ..., 9)
        
    The remaining 10 characters are non-alphanumeric:
        '!', '"', '&', "'", '(', ')', '-', '.', '?', '£'
        
    Where do these 10 non-alphanumeric characters appear in 
        the initial release of OmniPrint fine-grained alphabets?
        
    Answer:
    [   ('!', ['alphabets/fine/common_punctuations_symbols.txt']),
        ('"', []),
        ('&', ['alphabets/fine/common_punctuations_symbols.txt']),
        ("'", []),
        ('(', []),
        (')', []),
        ('-', []),
        ('.', []),
        ('?', ['alphabets/fine/common_punctuations_symbols.txt']),
        ('£', ['alphabets/fine/common_punctuations_symbols.txt'])]
        
    The number of instances of these 10 non-alphanumeric characters in 
        the training set (65 images in total):
            .    17
            !    10
            -     7
            '     7
            (     7
            )     7
            &     7
            "     1
            ?     1
            £     1
        
    The number of instances of these 10 non-alphanumeric characters in 
        the training set (45 images in total):
            .    11
            '     8
            !     8
            &     7
            -     4
            £     3
            ?     1
            "     1
            (     1
            )     1

    The training set size with only_alphanumeric=False and 
        only_keep_shared_classes=False: 6185
    The training set size with only_alphanumeric=False and 
        only_keep_shared_classes=True: 6178
    The training set size with only_alphanumeric=True and 
        only_keep_shared_classes=False: 6113
    The training set size with only_alphanumeric=True and 
        only_keep_shared_classes=True: 6113
    The training set size with 
        enforce_all_alphanumeric_classes=True: 6113

    The test set size with only_alphanumeric=False and 
        only_keep_shared_classes=False: 5430
    The test set size with only_alphanumeric=False and 
        only_keep_shared_classes=True: 5422
    The test set size with only_alphanumeric=True and 
        only_keep_shared_classes=False: 5379
    The test set size with only_alphanumeric=True and 
        only_keep_shared_classes=True: 5377
    The test set size with 
        enforce_all_alphanumeric_classes=True: 5379
    
    ========================================================
            
    train_split
        return the training dataset or the test dataset
    only_keep_test_set_classes
        return the cleaned dataset where only the classes contained in 
        the test set are kept. 
        Conflict with only_keep_shared_classes.
    only_keep_shared_classes
        return the cleaned dataset where only the classes shared among 
        the training set and test set are kept. 
        Conflict with only_keep_test_set_classes.
    only_alphanumeric
        return the cleaned dataset where only the alphanumeric classes 
        are kept
    enforce_all_alphanumeric_classes
        set only_alphanumeric=True, 
        set only_keep_test_set_classes=False,
        set only_keep_shared_classes=False,
        set self.classes = {all 26 uppercase basic latin, 
                            all 26 lowercase basic latin,
                            all 10 ASCII digits}
        even if training set does not have "z" examples
        
    only_keep_test_set_classes and only_keep_shared_classes cannot be 
    both True, only_alphanumeric can be used anywhere without conflicts.
    
    use_micro_version: if True, only use use 20 classes, these 20 classes 
        all have 20 examples per class at least in both original 
        training and original test split. use_micro_version overwrites 
        all other dataset split setting
    When use_micro_version is True, micro_split overwrites train_split. 
    """

    shared_classes_unicode = [33, 34, 38, 39, 40, 41, 45, 46, 48, 49,
                              50, 51, 52, 53, 54, 55, 56, 57, 63, 65,
                              66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
                              76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
                              86, 87, 88, 89, 90, 97, 98, 99, 100, 101,
                              102, 103, 104, 105, 106, 107, 108, 109,
                              110, 111, 112, 113, 114, 115, 116, 117,
                              118, 119, 120, 121, 163]

    alphanumeric = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67,
                    68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                    81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99,
                    100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                    110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
                    120, 121, 122]

    all_test_set_classes = [33, 34, 38, 39, 40, 41, 44, 45, 46, 48, 49,
                            50, 51, 52, 53, 54, 55, 56, 57, 63, 65, 66,
                            67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
                            78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
                            89, 90, 97, 98, 99, 100, 101, 102, 103, 104,
                            105, 106, 107, 108, 109, 110, 111, 112, 113,
                            114, 115, 116, 117, 118, 119, 120, 121, 122,
                            163]

    # ['0', '1', '2', 'B', 'D', 'E', 'I', 'K', 'L', 'P', 'W',
    # 'b', 'd', 'f', 'i', 'm', 'o', 'p', 's', 'v']
    micro_version_classes_unicode = [
        48, 49, 50, 66, 68, 69, 73, 75, 76, 80, 87, 98, 100, 102, 105,
        109, 111, 112, 115, 118]

    def __init__(self, train_split=True,
                 enforce_all_alphanumeric_classes=False,
                 only_keep_test_set_classes=False,
                 only_keep_shared_classes=False,
                 only_alphanumeric=False,
                 transform=None,
                 dataset_root="ICDAR_2003_character_dataset",
                 training_set_folder="TrialTrain_Set_6185_Characters",
                 test_set_folder="TrialTest_Set_5430_Characters",
                 use_micro_version=False,
                 micro_split="train",
                 use_albumentations=False):
        super().__init__()

        assert micro_split in ["train", "val", "test"]

        # enforce_all_alphanumeric_classes overrides attributes
        if enforce_all_alphanumeric_classes:
            only_alphanumeric = True
            only_keep_test_set_classes = False
            only_keep_shared_classes = False

        assert not (only_keep_test_set_classes is True and
                    only_keep_shared_classes is True), \
            "only_keep_test_set_classes and only_keep_shared_classes cannot be both True"

        self.train_split = train_split

        self.enforce_all_alphanumeric_classes = enforce_all_alphanumeric_classes
        self.only_keep_test_set_classes = only_keep_test_set_classes
        self.only_keep_shared_classes = only_keep_shared_classes
        self.only_alphanumeric = only_alphanumeric

        self.use_micro_version = use_micro_version
        self.micro_split = micro_split
        self.use_albumentations = use_albumentations

        if self.micro_split in ["train", "val"]:
            self.train_split = True
        elif self.micro_split in ["test"]:
            self.train_split = False

        self.transform = transform

        self.dataset_root = dataset_root
        self.training_set_folder = training_set_folder
        self.test_set_folder = test_set_folder

        if self.train_split:
            self.folder = os.path.join(
                self.dataset_root, self.training_set_folder)
        else:
            self.folder = os.path.join(
                self.dataset_root, self.test_set_folder)

        self.df = self.convert_ICDAR_2013_char_labels()
        self.set_classes_attributes()
        self.unicode2label = self.get_unicode2label_dict()
        self.items = self.df2items()

        if self.use_micro_version:
            # modifies self.items so that each class contains
            # 15/5/20 examples per class for train/val/test split
            self.prune_micro_examples()

        if self.use_albumentations:
            # modifies self.items
            self.transform_paths_to_opencv_images_in_RAM()

        self.classes_ = self.get_pytorch_classes_attributes()

    def set_classes_attributes(self):
        """
        self.classes contains a sorted list of classes 
        (in the form of rendered characters)
        """
        if self.use_micro_version:
            self.classes = sorted(self.df["raw_label"].unique().tolist())
        elif self.enforce_all_alphanumeric_classes:
            self.classes = [chr(x) for x in self.alphanumeric]
        else:
            self.classes = sorted(self.df["raw_label"].unique().tolist())

        self.n_classes = len(self.classes)
        self.classes_unicode = [ord(x) for x in self.classes]

    def get_pytorch_classes_attributes(self):
        """
        classes_ is usually called classes in torchvision datasets
        classes_ is a list, not numpy array nor torch tensor
        
        classes_ has the same length as the total number of examples
        """
        classes_ = []
        for idx in range(len(self.items)):
            X, y = self.items[idx]
            classes_.append(y)
        return classes_

    def get_unicode2label_dict(self):
        unicode2label = dict()
        for unicode_ in self.classes_unicode:
            if unicode_ not in unicode2label:
                unicode2label[unicode_] = len(unicode2label)
        return unicode2label

    def convert_ICDAR_2013_char_labels(self):
        """
        file: 
            char/1/1.jpg, char/1/2.jpg, char/1/3.jpg, ..., char/62/6181.jpg
        path:
            ICDAR_2003_character_dataset/TrialTrain_Set_6185_Characters/char/1/1.jpg,
            ICDAR_2003_character_dataset/TrialTrain_Set_6185_Characters/char/1/2.jpg, ...
        raw_label:
            s, e, l, f, a, ..., S, O, R, ..., 3, 1, 4, A, m, i, e, a, r
        raw_label_unicode:
            115, 101, 108, 102, 97, ...
        """
        df = []
        tree = ET.parse(os.path.join(self.folder, "char.xml"))
        root = tree.getroot()
        for i in range(len(root)):
            attrib_ = root[i].attrib
            df.append([attrib_["file"],
                       os.path.join(self.folder, attrib_["file"]),
                       attrib_["tag"]])
        df = pd.DataFrame(df, columns=["file", "path", "raw_label"])
        df["raw_label_unicode"] = df["raw_label"].apply(lambda x: ord(x))

        df = self.clean_classes(df)

        return df

    def clean_classes(self, df):
        if self.use_micro_version:
            df = df.drop(df[~df["raw_label_unicode"].isin(
                self.micro_version_classes_unicode)].index)
        else:
            if self.only_keep_test_set_classes:
                df = df.drop(df[~df["raw_label_unicode"].isin(
                    self.all_test_set_classes)].index)
            if self.only_keep_shared_classes:
                df = df.drop(df[~df["raw_label_unicode"].isin(
                    self.shared_classes_unicode)].index)
            if self.only_alphanumeric:
                df = df.drop(df[~df["raw_label_unicode"].isin(
                    self.alphanumeric)].index)
        df.reset_index(drop=True, inplace=True)
        return df

    def df2items(self):
        items = []
        for idx in range(self.df.shape[0]):
            img_path = self.df.loc[idx, "path"]
            target_unicode = self.df.loc[idx, "raw_label_unicode"]
            target = self.unicode2label[target_unicode]
            items.append([img_path, target])
        return items

    def prune_micro_examples(self,
                             train_examples_per_class=15,
                             val_examples_per_class=5,
                             test_examples_per_class=20):
        """
        modifies self.items so that each class contains 
        15/5/20 examples per class for train/val/test split
        """
        # reshuffle to break the order of examples or classes
        rng = np.random.default_rng(88)
        rng.shuffle(self.items)

        items_per_class = defaultdict(list)

        for img_path, target in self.items:
            items_per_class[target].append([img_path, target])

        pruned_items = []
        for class_ in items_per_class.keys():
            if self.micro_split == "train":
                pruned_items.extend(
                    items_per_class[class_][:train_examples_per_class])
            elif self.micro_split == "val":
                pruned_items.extend(
                    items_per_class[class_][train_examples_per_class:
                                            (train_examples_per_class + val_examples_per_class)])
            elif self.micro_split == "test":
                # micro test split is from the original test split, 
                # micro train, micro val splits are from the original train split, 
                # so it is OK to just take the first test_examples_per_class examples. 
                pruned_items.extend(
                    items_per_class[class_][:test_examples_per_class])

        # reshuffle to break the order of examples or classes
        rng = np.random.default_rng(66)
        rng.shuffle(pruned_items)

        self.items = pruned_items

    def transform_paths_to_opencv_images_in_RAM(self):
        """
        modifies self.items so that instead of storing image path 
        in RAM, we store OpenCV image in RAM (the dataset should 
        not be too large compared to RAM)
        """
        new_items = []
        for img_path, target in self.items:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            new_items.append([img, target])
        self.items = new_items

    def __getitem__(self, idx):
        x, target = self.items[idx]

        if not self.use_albumentations:
            # x is img_path
            img = Image.open(x)
            if self.transform is not None:
                img = self.transform(img)
        else:
            # x is img in OpenCV format
            img = self.transform(image=x)["image"]

        return img, target

    def __len__(self):
        return len(self.items)


class Benchmark40MicroSplitDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset with the field "targets"
    
    This dataset loads images into RAM at the initialization phase 
    in the form of OpenCV images, it is supposed to be used with 
    albumentations data augmentation library.
    
    https://github.com/albumentations-team/albumentations 
    """

    def __init__(self, dataset_directory, transform, split, 
                 train_examples_per_class=20, 
                 val_examples_per_class=0, 
                 test_examples_per_class=20):
        super().__init__()

        self.dataset_directory = dataset_directory

        assert transform is not None, "transform cannot be None"
        self.transform = transform

        assert split in ["train", "val", "test"]
        self.split = split
        
        self.train_examples_per_class = train_examples_per_class
        self.val_examples_per_class = val_examples_per_class
        self.test_examples_per_class = test_examples_per_class

        assert os.path.exists(dataset_directory), \
            "Dataset path {} not found.".format(dataset_directory)

        self.items = self.construct_items()
        self.divide_items_into_split()

        self.add_field_targets()

        self.n_classes = int(max(list(set(self.targets)))) + 1

        self.transform_paths_to_opencv_images_in_RAM()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img, target = self.items[idx]
        img = self.transform(image=img)["image"]
        return img, target

    def transform_paths_to_opencv_images_in_RAM(self):
        """
        modifies self.items in place so that instead of storing image path 
        in RAM, we store OpenCV image in RAM (the dataset should 
        not be too large compared to RAM)
        """
        new_items = []
        for img_path, target in self.items:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            new_items.append([img, target])
        self.items = new_items

    def divide_items_into_split(self):
        """
        modifies self.items in place so that each class contains 
        20/0/20 examples per class for train/val/test split
        """
        # reshuffle to break the order of examples or classes
        rng = np.random.default_rng(88)
        rng.shuffle(self.items)

        items_per_class = defaultdict(list)

        for img_path, target in self.items:
            items_per_class[target].append([img_path, target])

        pruned_items = []
        for class_ in items_per_class.keys():

            assert len(items_per_class[class_]) >= self.train_examples_per_class + \
                self.val_examples_per_class + self.test_examples_per_class

            if self.split == "train":
                pruned_items.extend(
                    items_per_class[class_][:self.train_examples_per_class])
            elif self.split == "val":
                pruned_items.extend(
                    items_per_class[class_][self.train_examples_per_class:
                                            (self.train_examples_per_class + self.val_examples_per_class)])
            elif self.split == "test":
                pruned_items.extend(
                    items_per_class[class_][(self.train_examples_per_class + self.val_examples_per_class):
                                            (self.train_examples_per_class + self.val_examples_per_class
                                                + self.test_examples_per_class)])
            else:
                raise Exception

        # reshuffle to break the order of examples or classes
        rng = np.random.default_rng(66)
        rng.shuffle(pruned_items)

        self.items = pruned_items

    def construct_items(self):
        """
        returns items: 
            a list of lists, each inner-list has 2 elements.
            the first is img_path, the second is the 
            classification label (integer).
        """
        # get raw labels
        self.read_info_json()
        df = self.read_labels_csv()

        tmp_items = []
        for idx_value, row in df.iterrows():
            img_path = os.path.join(
                self.dataset_directory, "images", idx_value)
            tmp_items.append([img_path,
                              df.loc[idx_value, self.category_column_name]])

        # map each raw label to a label (int starting from 0)
        self.raw_label2label = dict()
        items = []
        for item in tmp_items:
            if item[1] not in self.raw_label2label:
                self.raw_label2label[item[1]] = len(self.raw_label2label)
            items.append([item[0], self.raw_label2label[item[1]]])
        return items

    def read_info_json(self) -> None:
        info_json_path = os.path.join(
            self.dataset_directory, "info.json")
        with open(info_json_path, "r") as f:
            info_json = json.load(f)
        # "FILE_NAME"
        self.image_column_name = info_json["image_column_name"]
        # "CATEGORY"
        self.category_column_name = info_json["category_column_name"]

    def read_labels_csv(self):
        csv_path = os.path.join(
            self.dataset_directory, "labels.csv")
        df = pd.read_csv(csv_path, sep=",", encoding="utf-8")
        df = df.loc[:, [self.image_column_name, self.category_column_name]]
        df.set_index(self.image_column_name, inplace=True)
        return df

    def add_field_targets(self):
        """
        The targets field is available in nearly all torchvision datasets. 
        It must be a list containing the label for each data point (usually the y value).
        
        https://avalanche.continualai.org/how-tos/avalanchedataset/creating-avalanchedatasets
        """
        self.targets = []
        for item in self.items:
            self.targets.append(item[1])
        self.targets = torch.tensor(self.targets, dtype=torch.int64)
