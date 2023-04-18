import os
import sys

# in order to be able to import modules from packages in the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import shutil
import gc

import sklearn
import torch

import wandb

from shared_modules.sub_resnet152 import ResNet152Helper
from shared_modules.utils import *


def train(train_dataloader, model, device, epoch, 
          use_wandb, optim, scheduler, args, loss_kwargs, nb_classes):
    t0 = time.time()
    
    t_optim_cumul = 0

    train_loss = 0
    
    acc = AccuracyMetric(nb_classes=nb_classes)
    
    model.train()
    for batch_idx, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        
        t0_optim = time.time()

        # X.shape, y.shape
        # torch.Size([64, 3, 224, 224]) torch.Size([64])
        optim.zero_grad()
        
        logits = model(X)
        
        loss = compute_loss(args, logits, y, loss_kwargs)
        
        loss.backward()
        optim.step()
        
        if args.lr_scheduler in ["OneCycleLR"]:
            scheduler.step()
        if args.lr_scheduler in ["CosineAnnealingWarmRestarts"]:
            scheduler.step(epoch + batch_idx / len(train_dataloader))
        
        t_optim_cumul += time.time() - t0_optim
        
        train_loss += loss.item()
        acc.update(logits, y)


    train_loss /= len(train_dataloader)
    
    train_acc = acc.compute(mode="all")
    train_acc_balanced = acc.compute(mode="class_balanced")
    train_acc_worst = acc.compute(mode="worst_classes")
    
    t1 = time.time() - t0
    
    time_ratio_optim = t_optim_cumul / t1
    
    print("Epoch {} | Train loss {:.2f} | Train acc {:.2f} | Train acc (b) {:.2f} | Train acc (w) {:.2f} | Time {:.1f} seconds. Optimization time ratio: {:.2f}.".format(
        epoch+1, train_loss, train_acc, train_acc_balanced, 
        train_acc_worst, t1, time_ratio_optim))
    if use_wandb:
        wandb.log({"epoch": epoch+1, "train_loss": train_loss,
                   "train_acc": train_acc, "train_acc_balanced": train_acc_balanced,
                   "train_acc_worst": train_acc_worst,
                   "train_epoch_time": t1, 
                   "time_ratio_optim": time_ratio_optim})


def val(val_dataloader, model, device, epoch, use_wandb, 
        scheduler, args, best_val_record, loss_kwargs, nb_classes):
    
    t0 = time.time()

    val_loss = 0
    
    acc = AccuracyMetric(nb_classes=nb_classes)

    model.eval()
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(val_dataloader):
            X, y = X.to(device), y.to(device)
            
            logits = model(X)
            
            loss = compute_loss(args, logits, y, loss_kwargs)

            val_loss += loss.item()

            acc.update(logits, y)

    val_loss /= len(val_dataloader)
    
    val_acc = acc.compute(mode="all")
    val_acc_balanced = acc.compute(mode="class_balanced")
    val_acc_worst = acc.compute(mode="worst_classes")
    

    t1 = time.time() - t0
    
    if best_val_record[0] < val_acc_balanced:
        best_val_record[0] = val_acc_balanced
        best_val_record[1] = epoch + 1
    
    print("Epoch {} ({}) | Val loss {:.2f} | Val acc {:.2f} | Val acc (b) {:.2f} ({:.2f}) | Val acc (w) {:.2f} | Time {:.1f} seconds.".format(
        epoch+1, best_val_record[1], val_loss, val_acc, 
        val_acc_balanced, best_val_record[0], val_acc_worst, t1))
    
    
    if use_wandb:
        wandb.log({"epoch": epoch+1, "val_loss": val_loss,
                   "val_acc": val_acc, 
                   "val_acc_balanced": val_acc_balanced, 
                   "val_acc_worst": val_acc_worst,
                   "val_epoch_time": t1})
        
        wandb.run.summary["best_val_acc_balanced"] = best_val_record[0]
        wandb.run.summary["best_val_acc_balanced_epoch"] = best_val_record[1]
        
    return best_val_record


def test(val_dataloader, model, device, epoch, use_wandb,
        scheduler, args, best_val_record, loss_kwargs, nb_classes):

    t0 = time.time()

    val_loss = 0

    acc = AccuracyMetric(nb_classes=nb_classes)

    model.eval()
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(val_dataloader):
            X, y = X.to(device), y.to(device)

            logits = model(X)

            loss = compute_loss(args, logits, y, loss_kwargs)

            val_loss += loss.item()

            acc.update(logits, y)

    val_loss /= len(val_dataloader)

    val_acc = acc.compute(mode="all")
    val_acc_balanced = acc.compute(mode="class_balanced")
    val_acc_worst = acc.compute(mode="worst_classes")

    t1 = time.time() - t0

    print("Test loss {:.2f} | Test acc {:.2f} | Test acc (b) {:.2f} ({:.2f}) | Test acc (w) {:.2f} | Time {:.1f} seconds.".format(
        best_val_record[1], val_loss, val_acc,
        val_acc_balanced, best_val_record[0], val_acc_worst, t1))

    if use_wandb:
        wandb.run.summary["test_loss"] = val_loss
        wandb.run.summary["test_acc"] = val_acc
        wandb.run.summary["test_acc_balanced"] = val_acc_balanced
        wandb.run.summary["test_acc_worst"] = val_acc_worst
        wandb.run.summary["test_epoch_time"] = t1



def compute_class_weights(train_dataset, device):    
    classes = train_dataset.classes_

    # manually add "z" (Unicode 122 -> 61) because training set does not have it
    # this manual hack is OK only if enforce_all_alphanumeric_classes=True
    
    # this manual hack only distorts a little bit the true class distribution 
    # because it added one example of a non-existing class.
    classes.append(61)

    class_weights = sklearn.utils.class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(classes), y=np.asarray(classes))
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    class_weights = class_weights.to(device)
    return class_weights


def compute_loss_kwargs(args, train_dataset, device, nb_classes):
    loss_kwargs = dict()
    
    loss_kwargs["nb_classes"] = nb_classes
    
    if args.loss in ["weighted_cross_entropy", "weighted_focal_loss"]:
        loss_kwargs["class_weights"] = compute_class_weights(
            train_dataset, device)
    
    if args.loss in ["MSE_outlying_1", "MSE_outlying_5"]:
        loss_kwargs["class_support"] = compute_class_support(
            train_dataset, loss_kwargs["nb_classes"])
        
    return loss_kwargs


def initialize_with_full_resnet152(model, nb_classes):
    full_pretrained_model = timm.create_model("resnet152",
                                              pretrained=True,
                                              num_classes=nb_classes)
    missing_keys, unexpected_keys = model.load_state_dict(
        full_pretrained_model.state_dict(), strict=False)
    
    del full_pretrained_model
    gc.collect()
    
    return model


def initialize_new_model_using_the_old_one(args, model, 
                                           resnet152_less_blocks_checkpoint_dir, 
                                           use_wandb=False): 
    checkpoint_path = os.path.join(
        resnet152_less_blocks_checkpoint_dir, args.checkpoint_name)
    
    if args.nb_blocks != [1, 1, 1, 1]:
        assert os.path.exists(checkpoint_path)
        pretrained_model_state_dict = torch.load(checkpoint_path)
        
        model = load_weights_to_unmatched_model(
            model, pretrained_model_state_dict, verbose=True, use_wandb=use_wandb)
        
    if os.path.exists(resnet152_less_blocks_checkpoint_dir):
        shutil.rmtree(resnet152_less_blocks_checkpoint_dir)
    os.makedirs(resnet152_less_blocks_checkpoint_dir)
    
    return model, checkpoint_path


def parse_arguments():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="""how many subprocesses to use for data 
                        loading. 0 means that the data will be loaded 
                        in the main process.""")
    parser.add_argument("--no_wandb", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_scheduler", type=str, default="none", 
                        choices=["none", "OneCycleLR", "CosineAnnealingWarmRestarts"])
    parser.add_argument("--OneCycleLR_max_lr", type=float, default=3)
    parser.add_argument("--CosineAnnealingWarmRestarts_T_0", type=float, default=10)
    parser.add_argument("--CosineAnnealingWarmRestarts_T_mult", type=float, default=1)
    parser.add_argument("--optimizer", type=str, default="AdamW",
                        choices=["AdamW"])
    parser.add_argument("--training_set", type=str, 
                        default="full_train_split", 
                        choices=["full_train_split"], 
                        help="""Given that the test dataset is fixed, 
                        what training data to use? full_train_split means 
                        the whole original split of ICDAR 2003 character 
                        dataset.""")
    parser.add_argument("--data_augmentation", type=str,
                        default="RandAugmentIrttssStraugV1",
                        choices=["none", "RandAugmentIrttssStraugV1", 
                                 "RandAugmentIrttssV1"])
    parser.add_argument("--loss", type=str, default="cross_entropy",
                        choices=["cross_entropy", "weighted_cross_entropy", 
                                 "MSE", "MSE_outlying_1", "MSE_outlying_5",
                                 "focal_loss", "weighted_focal_loss"])
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--img_mode", type=str, default="RGB")
    
    parser.add_argument("--use_micro_version", action="store_true", default=False,
                        help="""use micro version of ICDAR dataset""")
    parser.add_argument("--use_albumentations", action="store_true", default=False,
                        help="""Load images into memory and use 
                        albumentations data aug for the training set""")
    
    # pick blocks from ResNet152
    parser.add_argument("--nb_blocks", type=str, default="1,1,1,1",
                        help="""comma-separated list of 4 integers. 
                        each element cannot surpass 3,8,36,3""")
    parser.add_argument("--checkpoint_name", type=str, default="less_blocks_checkpoint.pth")
    parser.add_argument("--use_ImageNet_pretraining", 
                        action="store_true", default=False)
    
    parser.add_argument("--paper_res", action="store_true", default=False)
    
    args = parser.parse_args()
    args.use_wandb = not args.no_wandb
    
    if args.paper_res:
        args.use_micro_version = True
        args.use_albumentations = True
    
    args.nb_blocks = [int(xx) for xx in args.nb_blocks.split(",")]
    assert len(args.nb_blocks) == 4
    assert args.nb_blocks[0] >= 1 and args.nb_blocks[0] <= 3
    assert args.nb_blocks[1] >= 1 and args.nb_blocks[1] <= 8
    assert args.nb_blocks[2] >= 1 and args.nb_blocks[2] <= 36
    assert args.nb_blocks[3] >= 1 and args.nb_blocks[3] <= 3
    
    if args.random_seed is not None:
        args.random_seed = args.random_seed + sum(args.nb_blocks)
    
    args.nb_blocks_in_total = sum(args.nb_blocks)
    
    return args


def init_wandb(args, verbose=True):
    project_name = "PickResnet152Blocks"
    group_name = "".format()
    
    wandb_dir = "wandb_logs"
    if not os.path.exists(wandb_dir):
        os.makedirs(wandb_dir)
    wandb.init(config=args, project=project_name,
               group=group_name, dir=wandb_dir)

    env_info = get_torch_gpu_environment()
    for k, v in env_info.items():
        wandb.run.summary[k] = v
    
    library_version = get_library_version()
    for k, v in library_version.items():
        wandb.run.summary[k] = v
    
    wandb_run_name = wandb.run.name
    
    if verbose:
        print("wandb_run_name: {}".format(wandb_run_name))
        print(env_info)
        print(library_version)
    return project_name, group_name, wandb_run_name



if __name__ == "__main__":
    t0_overall = time.time()
    
    args = parse_arguments()
    
    if args.use_wandb:
        project_name, group_name, wandb_run_name = init_wandb(args)
    else:
        print(get_torch_gpu_environment())
        print(get_library_version())
        wandb_run_name = ""
    
    set_random_seeds(random_seed=args.random_seed)
    
    device = get_device()
    
    ret = get_dataloaders(args)
    if len(ret) == 2:
        train_dataloader, test_dataloader = ret
        val_dataloader = None
    else:
        train_dataloader, val_dataloader, test_dataloader = ret
    
    nb_classes = train_dataloader.dataset.n_classes
    
    model = ResNet152Helper(
        nb_classes=nb_classes, divergence_idx=None,
        nb_branch=None, shift=0,
        nb_blocks=args.nb_blocks)
    
    if args.use_ImageNet_pretraining:
        # first, initialize the model with the monolithic full Resnet152
        model = initialize_with_full_resnet152(model, nb_classes)
    
    model.to(device)
    
    resnet152_less_blocks_checkpoint_dir = "resnet152_less_blocks_checkpoint_dir"
    
    # second, initialize (overwrite) parameters using the previous model
    model, checkpoint_path = initialize_new_model_using_the_old_one(
        args, model, resnet152_less_blocks_checkpoint_dir, 
        use_wandb=args.use_wandb)
    
    # retrain all parameters
    for name, param in model.named_parameters():
        param.requires_grad = True
        
    report_trainable_params_count(model, args.use_wandb)
    
    optim = get_optimizer(args, model)
    optim, scheduler = get_lr_scheduler(args, optim, train_dataloader)
    
    loss_kwargs = compute_loss_kwargs(
        args, train_dataloader.dataset, device, nb_classes)

    print("Training loop starts...")
    
    # the first is the best accuracy, the second is the epoch which achieved 
    # this accuracy, the third is the path to save model checkpoints
    dir_path = os.path.join("model_checkpoints")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    best_val_record = [- 1, - 1, 
                       os.path.join(dir_path, "{}.pth".format(
                           wandb_run_name))]

    for epoch in range(args.epochs): 
        train(train_dataloader, model, device, epoch,
              args.use_wandb, optim, scheduler, args, loss_kwargs, nb_classes)
        if epoch == args.epochs - 1:
            if val_dataloader is None:
                tmp = test_dataloader
            else:
                tmp = val_dataloader
            best_val_record = val(tmp, model, device, epoch,
                                args.use_wandb, scheduler, args, 
                                best_val_record, loss_kwargs, nb_classes)
    
    if val_dataloader is not None:
        test(test_dataloader, model, device, epoch,
                              args.use_wandb, scheduler, args, 
                              best_val_record, loss_kwargs, nb_classes)
    
    # save the model checkpoint for the next run
    print("Saving model to {}".format(checkpoint_path))
    torch.save(model.state_dict(), checkpoint_path)
    
    overall_time = time.time() - t0_overall
    
    print("Done in {:.2f} s.".format(overall_time))
    if args.use_wandb:
        wandb.run.summary["overall_time"] = overall_time
    

