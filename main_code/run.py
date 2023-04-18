import os
import sys
import copy
import warnings

# in order to be able to import modules from packages in the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import gc

import sklearn
import torch

import wandb
from torch_ema import ExponentialMovingAverage

from sub_resnet152 import get_diverged_resnet152_models, ConcatenatedModel
from torchensemble import FastGeometricClassifierMv2
from utils import *


def train(train_dataloader, model, device, epoch, 
          optim, args, loss_kwargs, model_idx, ema_list):
    t0 = time.time()
    
    t_optim_cumul = 0

    train_loss = 0
    
    acc = QuickAccuracyMetric()
    
    model.train()
    for X, y in train_dataloader:
        X, y = X.to(device), y.to(device)
        
        t0_optim = time.time()

        # X.shape, y.shape
        # torch.Size([32, 3, 128, 128]) torch.Size([32])
        optim.zero_grad()
        
        logits = model(X)
        
        loss = compute_loss(args, logits, y, loss_kwargs)
        
        loss.backward()
        optim.step()
        
        t_optim_cumul += time.time() - t0_optim
        
        # here, removed lr_scheduler check for OneCycleLR and CosineAnnealingWarmRestarts
        
        if ema_list is not None:
            # Update the moving average with the new 
            # parameters from the last optimizer step
            ema_list[model_idx].update()
        
        train_loss += loss.item()
        acc.update(logits, y)


    train_loss /= len(train_dataloader)
    
    train_acc = acc.compute()
    
    t1 = time.time() - t0
    
    time_ratio_optim = t_optim_cumul / t1
    
    train_result_dict = dict()
    train_result_dict["epoch"] = epoch + 1
    train_result_dict["train_loss"] = train_loss
    train_result_dict["train_acc"] = train_acc
    train_result_dict["train_epoch_time"] = t1
    train_result_dict["true_optim_time"] = t_optim_cumul
    train_result_dict["time_ratio_optim"] = time_ratio_optim
    return train_result_dict


def aggregate_train_results(train_result_dict_list, args):
    cumul_true_optim_time = 0
    train_loss = 0
    train_acc = 0
    train_epoch_time = 0
    for idx in range(len(train_result_dict_list)):
        epoch = train_result_dict_list[idx]["epoch"]
        train_loss += train_result_dict_list[idx]["train_loss"]
        train_acc += train_result_dict_list[idx]["train_acc"]
        train_epoch_time += train_result_dict_list[idx]["train_epoch_time"]
        cumul_true_optim_time += train_result_dict_list[idx]["true_optim_time"]
    
    train_loss /= len(train_result_dict_list)
    train_acc /= len(train_result_dict_list)
    time_ratio_optim = cumul_true_optim_time / train_epoch_time
    train_epoch_time /= len(train_result_dict_list)
    true_optim_time = cumul_true_optim_time / len(train_result_dict_list)
    
    print("Epoch {} | Train loss {:.2f} | Train acc {:.2f} | Time {:.1f} ({:.1f}) seconds. Optimization time ratio: {:.2f}.".format(
        epoch, train_loss, train_acc, train_epoch_time, true_optim_time, time_ratio_optim))
    if args.use_wandb:
        wandb.log({"epoch": epoch, "train_loss": train_loss,
                   "train_acc": train_acc, 
                   "train_epoch_time": train_epoch_time,
                   "time_ratio_optim": time_ratio_optim,
                   "true_optim_time": true_optim_time})


def branches_inference(base_models, X):
    intermediate_feature = base_models[0].stem(X)
    proba_list = []
    for model in base_models:
        logits = model.branch(intermediate_feature)
        proba = F.softmax(logits, dim=1)
        proba_list.append(proba)

    # averaged_proba has the same shape as any element in proba_list
    averaged_proba = sum(proba_list) / len(proba_list)
    
    return averaged_proba


def baseline_inference(base_models, X):
    model = base_models[0]
    logits = model(X)
    proba = F.softmax(logits, dim=1)
    return proba


def inference_one_batch(base_models, X, ema_list, ema_restore=True):
    """
    ema_restore should be True if training and validation interveins 
    and training is not using EMA. 
    
    If the test is only done at the very end, then one can 
    set ema_restore=False. 
    """
    if ema_list is not None:
        # manually do the job of average_parameters context manager
        for model_idx in range(len(base_models)):
            # Save the current parameters for restoring later
            ema_list[model_idx].store()
            # Copy current averaged parameters into given collection of parameters
            ema_list[model_idx].copy_to()
    
    if isinstance(base_models[0], ConcatenatedModel):
        # stem and branches
        proba = branches_inference(base_models, X)
    else:
        # baseline
        proba = baseline_inference(base_models, X)
        
    if ema_list is not None:
        if ema_restore:
            # manually do the job of average_parameters context manager
            for model_idx in range(len(base_models)):
                # Restore the parameters stored with the "store" method
                ema_list[model_idx].restore()
    
    return proba

def val(val_dataloader, base_models, device, epoch,
        args, best_val_record, ema_list):
    
    t0 = time.time()
    
    acc = QuickAccuracyMetric()
    
    ema_restore = True

    for model in base_models:
        model.eval()
    with torch.no_grad():
        for X, y in val_dataloader:
            X, y = X.to(device), y.to(device)
            
            proba = inference_one_batch(base_models, X, ema_list, 
                                        ema_restore=ema_restore)
            
            acc.update(proba, y)
    
    val_acc = acc.compute()
    
    t1 = time.time() - t0
    
    if best_val_record[0] < val_acc:
        best_val_record[0] = val_acc
        best_val_record[1] = epoch + 1
    
    print("Epoch {} ({}) | Val acc {:.2f} ({:.2f}) | Time {:.1f} seconds.".format(
        epoch+1, best_val_record[1], val_acc, best_val_record[0], t1))
    
    if args.use_wandb:
        wandb.log({"epoch": epoch+1, 
                   "val_acc": val_acc, 
                   "val_epoch_time": t1})
        
        wandb.run.summary["best_val_acc"] = best_val_record[0]
        wandb.run.summary["best_val_acc_epoch"] = best_val_record[1]
        
    return best_val_record


def test(test_dataloader, base_models, device, args, ema_list):

    t0 = time.time()

    acc = QuickAccuracyMetric()
    
    ema_restore = False

    for model in base_models:
        model.eval()
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)

            proba = inference_one_batch(base_models, X, ema_list,
                                        ema_restore=ema_restore)

            acc.update(proba, y)

    test_acc = acc.compute()

    t1 = time.time() - t0

    print("Test acc {:.2f} | Time {:.1f} seconds.".format(test_acc, t1))

    if args.use_wandb:
        wandb.run.summary["test_acc"] = test_acc
        wandb.run.summary["test_epoch_time"] = t1


def get_optim_and_scheduler(args, model, train_dataloader):
    optim = get_optimizer(args, model)
    optim, scheduler = get_lr_scheduler(args, optim, train_dataloader)
    return optim, scheduler


def build_ema_list(args, epoch, base_models):
    ema_list = None
    if args.ema_epochs is not None:
        ema_start_epoch = args.epochs - args.ema_epochs
        if epoch == ema_start_epoch: 
            ema_list = []
            # create one ema object for each branch
            for model_idx in range(len(base_models)):
                model = base_models[model_idx]
                # only trainable parameters (branch) are tracked by EMA
                ema = ExponentialMovingAverage(
                    parameters=[p for p in model.parameters()
                                if p.requires_grad],
                    decay=args.ema_decay)
                ema_list.append(ema)
            print("Start using EMA at epoch {}/{}".format(epoch + 1, args.epochs))
    
    return ema_list

def train_val_test(args, base_models, train_dataloader, val_dataloader,
                   test_dataloader, wandb_run_name, device, loss_kwargs):
    # set up optimizers for each branch
    optim_list = []
    scheduler_list = []
    for model_idx in range(len(base_models)):
        optim, scheduler = get_optim_and_scheduler(
            args, base_models[model_idx], train_dataloader)
        optim_list.append(optim)
        scheduler_list.append(scheduler)
    
    print("Training loop starts...")

    # the first is the best accuracy, the second is the epoch which achieved
    # this accuracy, the third is the directory to save model checkpoints, 
    # the fourth is the wandb run name
    dir_path = os.path.join("model_checkpoints")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    best_val_record = [- 1, - 1, dir_path, wandb_run_name]

    ema_list = None
    
    for epoch in range(args.epochs):
        # for the last X training epochs,
        # start using EMA to build val/test model
        ema_list = build_ema_list(args, epoch, base_models)
        
        train_result_dict_list = []
        for model_idx in range(len(base_models)):
            
            train_result_dict = train(train_dataloader, 
                                      base_models[model_idx], 
                                      device, epoch, 
                                      optim_list[model_idx], 
                                      args, loss_kwargs, 
                                      model_idx, ema_list)
            
            train_result_dict_list.append(train_result_dict)

        aggregate_train_results(train_result_dict_list, args)

        if args.do_validation:
            best_val_record = val(val_dataloader, base_models, device, epoch,
                                  args, best_val_record, ema_list)

    
    test(test_dataloader, base_models, device, args, ema_list)
    

def train_test_fge(args, base_models, train_dataloader, 
                   test_dataloader, device, nb_classes, wandb_run_name):
    
    def _branches_inference(base_models, X):
        intermediate_feature = base_models[0].stem(X)
        logits_list = []
        for model in base_models:
            logits = model.branch(intermediate_feature)
            logits_list.append(logits)
        
        averaged_logits = sum(logits_list) / len(logits_list)
        proba = F.softmax(averaged_logits, dim=1)
        return proba
    
    def _test(test_dataloader, base_models, device, args):
        t0 = time.time()

        acc = QuickAccuracyMetric()

        for model in base_models:
            model.eval()
        with torch.no_grad():
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)

                proba = _branches_inference(base_models, X)

                acc.update(proba, y)

        test_acc = acc.compute()

        t1 = time.time() - t0

        print("Test acc {:.2f} | Time {:.1f} seconds.".format(test_acc, t1))

        if args.use_wandb:
            wandb.run.summary["test_acc"] = test_acc
            wandb.run.summary["test_epoch_time"] = t1

    model = FastGeometricClassifierMv2(
        estimators=base_models,
        device=device,
        nb_classes=nb_classes,
        use_wandb=args.use_wandb
    )

    if args.loss == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    else:
        raise Exception("Loss {} not supported for FGE.".format(args.loss))
    model.set_criterion(criterion)

    if args.optimizer == "AdamW":
        model.set_optimizer('AdamW', lr=args.lr,
                            weight_decay=args.weight_decay)
    else:
        raise Exception("")
    
    dir_path = os.path.join("model_checkpoints")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    model.fit(
        train_dataloader,
        epochs=args.epochs,
        test_loader=None,
        save_model=False,
        save_dir=os.path.join(dir_path, wandb_run_name),
        lr_1=args.fge_lr_1,
        lr_2=args.fge_lr_2
    )
    
    _test(test_dataloader, model.estimators_, device, args)
    

def get_baseline_model(args, nb_classes, device):
    model = timm.create_model(args.baseline_model,
                              pretrained=True,
                              num_classes=nb_classes)
    model.default_cfg["input_size"] = (3, args.img_size, args.img_size)
    
    if args.train_layers == "only_last":
        for name, param in model.named_parameters():
            if name.startswith(model.default_cfg["classifier"]):
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif args.train_layers in ["retrain_all"]:
        for name, param in model.named_parameters():
            param.requires_grad = True
    
    model.to(device)
    return model


def record_model_attributes(model, use_wandb=False, verbose=False):
    """
    model: ConcatenatedModel
    """
    attributes = dict()
    attributes["planes_per_block"] = model.planes_per_block
    attributes["diverged_planes_per_block"] = model.diverged_planes_per_block
    attributes["shift"] = model.shift

    if verbose:
        print(attributes)
    if use_wandb:
        for k, v in attributes.items():
            wandb.run.summary[k] = v
    return


def compute_class_weights(train_dataset, device):    
    # classes has the same length as the total number of examples
    classes = train_dataset.classes_
    
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


def parse_arguments():
    import argparse
    
    parser = argparse.ArgumentParser()
    
    # general-purpose
    parser.add_argument("--wandb_project_name", type=str, default=None)
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="""how many subprocesses to use for data 
                        loading. 0 means that the data will be loaded 
                        in the main process.""")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_scheduler", type=str, default="none", 
                        choices=["none"])
    parser.add_argument("--optimizer", type=str, default="AdamW",
                        choices=["AdamW"])
    parser.add_argument("--loss", type=str, default="cross_entropy",
                        choices=["cross_entropy"])
    parser.add_argument("--img_size", type=int, default=128)
    
    # general-purpose binary switch
    parser.add_argument("--no_wandb", action="store_true", default=False)
    parser.add_argument("--do_validation", action="store_true", default=False, 
                        help="""do not use this if there is not validation split""")
    
    # dataset
    parser.add_argument("--dataset", type=str, default="icdar03_char_micro", 
                        help="""icdar03_char_micro, BCT_Micro, BRD_Micro, 
                        CRS_Micro, etc.""")
    parser.add_argument("--train_examples_per_class", type=int, default=20, 
                        help="Not used for icdar03_char_micro")
    parser.add_argument("--val_examples_per_class", type=int, default=0, 
                        help="Not used for icdar03_char_micro")
    parser.add_argument("--test_examples_per_class", type=int, default=20, 
                        help="Not used for icdar03_char_micro")
    
    # split resnet152
    parser.add_argument("--divergence_idx", type=int, default=15)
    parser.add_argument("--nb_branch", type=int, default=2)
    parser.add_argument("--split_pretrained_weights",
                        action="store_true", default=False)
    parser.add_argument("--warm_stem_epochs", type=int, default=10)
    
    
    parser.add_argument("--nb_blocks", type=str, default="1,1,1,1",
                        help="""comma-separated list of 4 integers. 
                        each element cannot surpass 3,8,36,3""")
    parser.add_argument("--pretrained_model", type=str, default="resnet152", 
                        choices=["resnet152"])
    
    # EMA
    parser.add_argument("--ema_epochs", type=int, default=None,
                        help="""ema_epochs=None means not using EMA 
                        for validation and test. Otherwise, if ema_epochs=X, 
                        then EMA will be run for the last X epochs of 
                        training. Training model does not use EMA, EMA 
                        model copy is only used for validation and test.""")
    
    # baseline
    parser.add_argument("--run_baseline", action="store_true", default=False,
                        help="""Run usual baseline instead of branches""")
    parser.add_argument("--baseline_model", type=str, default="resnet152", 
                        choices=["resnet152"])
    parser.add_argument("--train_layers", type=str, default="only_last", 
                        choices=["only_last", "retrain_all"],
                        help="How to train baseline. Only used if run_baseline=True.")
    
    # FGE
    parser.add_argument("--run_FGE", action="store_true", default=False)
    parser.add_argument("--fge_lr_1", type=float, default=1e-3,
                        help="""Used by Fast Geometric Ensemble. 
                        alpha_1 in original paper used to adjust 
                        the learning rate, also serves as the 
                        initial learning rate of the internal 
                        optimizer.""")
    parser.add_argument("--fge_lr_2", type=float, default=5e-4,
                        help="""Used by Fast Geometric Ensemble. 
                        alpha_2 in original paper used to adjust 
                        the learning rate, also serves as the 
                        smallest learning rate of the internal 
                        optimizer.""")
    
    args = parser.parse_args()
    args.use_wandb = not args.no_wandb
    
    args.nb_blocks = [int(xx) for xx in args.nb_blocks.split(",")]
    assert len(args.nb_blocks) == 4
    assert args.nb_blocks[0] >= 1 and args.nb_blocks[0] <= 3
    assert args.nb_blocks[1] >= 1 and args.nb_blocks[1] <= 8
    assert args.nb_blocks[2] >= 1 and args.nb_blocks[2] <= 36
    assert args.nb_blocks[3] >= 1 and args.nb_blocks[3] <= 3
    
    if args.random_seed is not None:
        # related to sum(args.blocks = [1, 1, 1, 1])
        args.random_seed = args.random_seed + 4
    
    # automatically compute EMA decay rate
    if args.ema_epochs is not None:
        # ema_epochs = 1 / (1 - ema_decay)
        # ==> ema_decay = (ema_epochs - 1) / ema_epochs
        assert args.ema_epochs >= 0 and args.ema_epochs <= args.epochs
        args.ema_decay = (args.ema_epochs - 1) / args.ema_epochs
        
    assert not (args.run_baseline and args.run_FGE)
    if args.run_FGE and args.do_validation:
        warnings.warn("""run_FGE activated, 
                      set do_validation to False.""".format(args.dataset))
        args.do_validation = False
    
    return args


def init_wandb(args, verbose=True):
    if args.wandb_project_name is None:
        project_name = "RRR_1_EMA_{}".format(args.dataset)
    else:
        project_name = args.wandb_project_name

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
    
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(args)
    if val_dataloader is None and args.do_validation:
        warnings.warn("""Dataset {} does not have validation 
                      split, set do_validation to False.""".format(args.dataset))
        args.do_validation = False
    
    nb_classes = train_dataloader.dataset.n_classes
    if args.use_wandb:
        wandb.run.summary["nb_classes"] = nb_classes
    
    if not args.run_baseline:
        base_models = get_diverged_resnet152_models(divergence_idx=args.divergence_idx,
                                                    nb_branch=args.nb_branch,
                                                    nb_classes=nb_classes,
                                                    split_pretrained_weights=args.split_pretrained_weights,
                                                    device=device, concat=True,
                                                    train_dataloader=train_dataloader,
                                                    warm_stem_epochs=args.warm_stem_epochs,
                                                    pretrained_model=args.pretrained_model,
                                                    nb_blocks=args.nb_blocks)

        record_model_attributes(
            base_models[0], use_wandb=args.use_wandb, verbose=True)

        for base_model in base_models:
            base_model.freeze_stem_train_branch()
    else:
        # create baseline model
        model = get_baseline_model(args, nb_classes, device)
        base_models = [model]
    
    report_trainable_params_count(base_models[0], args.use_wandb)
    
    loss_kwargs = compute_loss_kwargs(
        args, train_dataloader.dataset, device, nb_classes)
    
    # main work here
    if args.run_FGE:
        train_test_fge(args, base_models, train_dataloader,
                       test_dataloader, device, nb_classes, wandb_run_name)
    else:
        train_val_test(args, base_models, train_dataloader, val_dataloader,
                       test_dataloader, wandb_run_name, device, loss_kwargs)

    
    overall_time = time.time() - t0_overall
    
    print("Done in {:.2f} s.".format(overall_time))
    if args.use_wandb:
        wandb.run.summary["overall_time"] = overall_time
    

