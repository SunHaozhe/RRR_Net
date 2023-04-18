"""
  In bagging-based ensemble, each base estimator is trained independently.
  In addition, sampling with replacement is conducted on the training data
  batches to encourage the diversity between different base estimators in
  the ensemble.
"""

import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
from joblib import Parallel, delayed

from ._base import BaseClassifier, BaseRegressor
from ._base import torchensemble_model_doc
from .utils import io
from .utils import set_module
from .utils import operator as op

import wandb
from .utils.hz import *


__all__ = ["BaggingClassifier", "BaggingRegressor"]


def _parallel_fit_per_epoch(
    train_loader,
    estimator,
    cur_lr,
    optimizer,
    criterion,
    idx,
    epoch,
    log_interval,
    device,
    is_classification,
    nb_classes
):
    """
    Private function used to fit base estimators in parallel.

    WARNING: Parallelization when fitting large base estimators may cause
    out-of-memory error.
    """
    if cur_lr:
        # Parallelization corrupts the binding between optimizer and scheduler
        set_module.update_lr(optimizer, cur_lr)
        
    acc = AccuracyMetric(nb_classes=nb_classes)
    train_loss = 0
    
    t_optim_cumul = 0
    t0 = time.time()
    
    for batch_idx, elem in enumerate(train_loader):

        data, target = io.split_data_target(elem, device)
        batch_size = data[0].size(0)

        # Sampling with replacement
        sampling_mask = torch.randint(
            high=batch_size, size=(int(batch_size),), dtype=torch.int64
        )
        sampling_mask = torch.unique(sampling_mask)  # remove duplicates
        subsample_size = sampling_mask.size(0)
        sampling_data = [tensor[sampling_mask] for tensor in data]
        sampling_target = target[sampling_mask]

        t0_optim = time.time()
        
        optimizer.zero_grad()
        sampling_output = estimator(*sampling_data)
        loss = criterion(sampling_output, sampling_target)
        loss.backward()
        optimizer.step()
        
        t_optim_cumul += time.time() - t0_optim
        
        train_loss += loss.item()
        acc.update(sampling_output, sampling_target)

    t1 = time.time() - t0
    time_ratio_optim = t_optim_cumul / t1
    
    train_loss /= len(train_loader)
    
    return estimator, optimizer, train_loss, acc, time_ratio_optim


@torchensemble_model_doc(
    """Implementation on the BaggingClassifier.""", "model"
)
class BaggingClassifier(BaseClassifier):
    @torchensemble_model_doc(
        """Implementation on the data forwarding in BaggingClassifier.""",
        "classifier_forward",
    )
    def forward(self, *x):
        # Average over class distributions from all base estimators.
        outputs = [
            F.softmax(estimator(*x), dim=1) for estimator in self.estimators_
        ]
        proba = op.average(outputs)

        return proba

    @torchensemble_model_doc(
        """Set the attributes on optimizer for BaggingClassifier.""",
        "set_optimizer",
    )
    def set_optimizer(self, optimizer_name, **kwargs):
        super().set_optimizer(optimizer_name, **kwargs)

    @torchensemble_model_doc(
        """Set the attributes on scheduler for BaggingClassifier.""",
        "set_scheduler",
    )
    def set_scheduler(self, scheduler_name, **kwargs):
        super().set_scheduler(scheduler_name, **kwargs)

    @torchensemble_model_doc(
        """Set the training criterion for BaggingClassifier.""",
        "set_criterion",
    )
    def set_criterion(self, criterion):
        super().set_criterion(criterion)

    @torchensemble_model_doc(
        """Implementation on the training stage of BaggingClassifier.""", "fit"
    )
    def fit(
        self,
        train_loader,
        epochs=100,
        log_interval=100,
        test_loader=None,
        save_model=True,
        save_dir=None,
    ):

        self._validate_parameters(epochs, log_interval)

        # Instantiate a pool of optimizers, and schedulers.
        optimizers = []
        for i in range(self.n_estimators):
            optimizers.append(
                set_module.set_optimizer(
                    self.estimators_[i], self.optimizer_name, **self.optimizer_args
                )
            )

            if self.use_scheduler_:
                scheduler_ = set_module.set_scheduler(
                    optimizers[i], self.scheduler_name, **self.scheduler_args
                )

        # Check the training criterion
        if not hasattr(self, "_criterion"):
            self._criterion = nn.CrossEntropyLoss()

        # Utils
        best_acc = 0.0
        best_acc_epoch = -1

        # Internal helper function on pesudo forward
        def _forward(estimators, *x):
            outputs = [
                F.softmax(estimator(*x), dim=1) for estimator in estimators
            ]
            proba = op.average(outputs)

            return proba
        
        
        # Maintain a pool of workers
        with Parallel(n_jobs=self.n_jobs) as parallel:

            # Training loop
            for epoch in range(epochs):
                self.train()

                if self.use_scheduler_:
                    cur_lr = scheduler_.get_last_lr()[0]
                else:
                    cur_lr = None

                if self.n_jobs and self.n_jobs > 1:
                    msg = "Parallelization on the training epoch: {}"
                    print(msg.format(epoch + 1))

                t0 = time.time()
                
                rets = parallel(
                    delayed(_parallel_fit_per_epoch)(
                        train_loader,
                        estimator,
                        cur_lr,
                        optimizer,
                        self._criterion,
                        idx,
                        epoch,
                        log_interval,
                        self.device,
                        True,
                        self.n_outputs
                    )
                    for idx, (estimator, optimizer) in enumerate(
                        zip(self.estimators_, optimizers)
                    )
                )

                estimators, optimizers = [], []
                losses, accs = [], []
                time_ratio_optim_list = []
                for estimator, optimizer, train_loss, acc, time_ratio_optim in rets:
                    estimators.append(estimator)
                    optimizers.append(optimizer)
                    losses.append(train_loss)
                    accs.append(acc)
                    time_ratio_optim_list.append(time_ratio_optim)
                    
                t1 = time.time() - t0
                    
                # average across base models
                train_loss = np.mean(losses) 
                acc_all_list, acc_balanced_list, acc_worst_list = [], [], []
                for acc in accs:
                    acc_all_list.append(acc.compute(mode="all"))
                    acc_balanced_list.append(acc.compute(mode="class_balanced"))
                    acc_worst_list.append(acc.compute(mode="worst_classes"))
                acc_ = np.mean(acc_all_list)
                acc_balanced_ = np.mean(acc_balanced_list)
                acc_worst_ = np.mean(acc_worst_list)
                time_ratio_optim = np.mean(time_ratio_optim_list)
                
                print("Epoch {} | Train loss {:.2f} | Train acc {:.2f} | Train acc (b) {:.2f} | Train acc (w) {:.2f} | Time {:.1f} seconds. Optimization time ratio: {:.2f}.".format(
                    epoch+1, train_loss, acc_, acc_balanced_,
                    acc_worst_, t1, time_ratio_optim))
                if self.use_wandb:
                    wandb.log({"epoch": epoch+1, "train_loss": train_loss,
                               "train_acc": acc_, "train_acc_balanced": acc_balanced_,
                               "train_acc_worst": acc_worst_,
                               "train_epoch_time": t1, 
                               "time_ratio_optim": time_ratio_optim})
                
                self.estimators_ = nn.ModuleList()
                self.estimators_.extend(estimators)

                # Validation
                if test_loader:
                    self.eval()
                    with torch.no_grad():
                        t0 = time.time()
                        val_loss = 0
                        acc = AccuracyMetric(nb_classes=self.n_outputs)
                        for _, elem in enumerate(test_loader):
                            data, target = io.split_data_target(
                                elem, self.device
                            )
                            output = _forward(self.estimators_, *data)
                            
                            loss = self._criterion(output, target)
                            
                            val_loss += loss.item()
                            acc.update(output, target)
                        
                        t1 = time.time() - t0    
                        
                        val_loss /= len(test_loader)
                        val_acc = acc.compute(mode="all")
                        val_acc_balanced = acc.compute(mode="class_balanced")
                        val_acc_worst = acc.compute(mode="worst_classes")
                        
                        if val_acc_balanced > best_acc:
                            best_acc = val_acc_balanced
                            best_acc_epoch = epoch + 1
                            if save_model:
                                io.save(self, save_dir)
                        
                        print("Epoch {} ({}) | Val loss {:.2f} | Val acc {:.2f} | Val acc (b) {:.2f} ({:.2f}) | Val acc (w) {:.2f} | Time {:.1f} seconds.".format(
                            epoch+1, best_acc_epoch, val_loss, val_acc,
                            val_acc_balanced, best_acc, val_acc_worst, t1))

                        if self.use_wandb:
                            wandb.log({"epoch": epoch+1, "val_loss": val_loss,
                                    "val_acc": val_acc,
                                    "val_acc_balanced": val_acc_balanced,
                                    "val_acc_worst": val_acc_worst,
                                    "val_epoch_time": t1})

                            wandb.run.summary["best_val_acc_balanced"] = best_acc
                            wandb.run.summary["best_val_acc_balanced_epoch"] = best_acc_epoch

                # Update the scheduler
                with warnings.catch_warnings():

                    # UserWarning raised by PyTorch is ignored because
                    # scheduler does not have a real effect on the optimizer.
                    warnings.simplefilter("ignore", UserWarning)

                    if self.use_scheduler_:
                        scheduler_.step()

        
        if save_model and not test_loader:
            io.save(self, save_dir)

    @torchensemble_model_doc(item="classifier_evaluate")
    def evaluate(self, test_loader, return_loss=False):
        return super().evaluate(test_loader, return_loss)

    @torchensemble_model_doc(item="predict")
    def predict(self, *x):
        return super().predict(*x)


class BaggingClassifierM(BaggingClassifier):
    """
    sampling with replacement at the very 
    beginning of the fit method, which is 
    to use the sampling with replacement 
    to create N dataloaders/datasets 
    (assume that there are N base models 
    to learn), each of the N 
    dataloaders/datasets can have duplicates. 
    Then in the function _parallel_fit_per_epoch, 
    the data batch are used in the classic 
    way without further subsampling.
    
    M stands for "modified" or "my"
    """
    def fit(
        self,
        train_loader,
        epochs=100,
        log_interval=100,
        test_loader=None,
        save_model=True,
        save_dir=None,
    ):

        self._validate_parameters(epochs, log_interval)

        # Instantiate a pool of optimizers, and schedulers.
        optimizers = []
        for i in range(self.n_estimators):
            optimizers.append(
                set_module.set_optimizer(
                    self.estimators_[i], self.optimizer_name, **self.optimizer_args
                )
            )

            if self.use_scheduler_:
                scheduler_ = set_module.set_scheduler(
                    optimizers[i], self.scheduler_name, **self.scheduler_args
                )

        # Check the training criterion
        if not hasattr(self, "_criterion"):
            self._criterion = nn.CrossEntropyLoss()

        # Utils
        best_acc = 0.0
        best_acc_epoch = -1

        # Internal helper function on pesudo forward
        def _forward(estimators, *x):
            outputs = [
                F.softmax(estimator(*x), dim=1) for estimator in estimators
            ]
            proba = op.average(outputs)

            return proba
        
        def _forward_eval(estimators, *x):
            stem = estimators[0].stem
            intermediate_feature = stem(*x)
            outputs = [
                F.softmax(estimator.branch(intermediate_feature), 
                          dim=1) for estimator in estimators
            ]
            proba = op.average(outputs)

            return proba

        # Maintain a pool of workers
        with Parallel(n_jobs=self.n_jobs) as parallel:

            # Training loop
            for epoch in range(epochs):
                self.train()

                if self.use_scheduler_:
                    cur_lr = scheduler_.get_last_lr()[0]
                else:
                    cur_lr = None

                if self.n_jobs and self.n_jobs > 1:
                    msg = "Parallelization on the training epoch: {}"
                    print(msg.format(epoch + 1))

                t0 = time.time()

                rets = parallel(
                    delayed(self.parallel_fit_per_epoch)(
                        dataloader,
                        estimator,
                        cur_lr,
                        optimizer,
                        self._criterion,
                        idx,
                        epoch,
                        log_interval,
                        self.device,
                        True,
                        self.n_outputs
                    )
                    for idx, (estimator, optimizer, dataloader) in enumerate(
                        zip(self.estimators_, optimizers, train_loader)
                    )
                )

                estimators, optimizers = [], []
                losses, accs = [], []
                time_ratio_optim_list = []
                for estimator, optimizer, train_loss, acc, time_ratio_optim in rets:
                    estimators.append(estimator)
                    optimizers.append(optimizer)
                    losses.append(train_loss)
                    accs.append(acc)
                    time_ratio_optim_list.append(time_ratio_optim)

                t1 = time.time() - t0

                # average across base models
                train_loss = np.mean(losses)
                acc_all_list, acc_balanced_list, acc_worst_list = [], [], []
                for acc in accs:
                    acc_all_list.append(acc.compute(mode="all"))
                    acc_balanced_list.append(
                        acc.compute(mode="class_balanced"))
                    acc_worst_list.append(acc.compute(mode="worst_classes"))
                acc_ = np.mean(acc_all_list)
                acc_balanced_ = np.mean(acc_balanced_list)
                acc_worst_ = np.mean(acc_worst_list)
                time_ratio_optim = np.mean(time_ratio_optim_list)

                print("Epoch {} | Train loss {:.2f} | Train acc {:.2f} | Train acc (b) {:.2f} | Train acc (w) {:.2f} | Time {:.1f} seconds. Optimization time ratio: {:.2f}.".format(
                    epoch+1, train_loss, acc_, acc_balanced_,
                    acc_worst_, t1, time_ratio_optim))
                if self.use_wandb:
                    wandb.log({"epoch": epoch+1, "train_loss": train_loss,
                               "train_acc": acc_, "train_acc_balanced": acc_balanced_,
                               "train_acc_worst": acc_worst_,
                               "train_epoch_time": t1,
                               "time_ratio_optim": time_ratio_optim})

                self.estimators_ = nn.ModuleList()
                self.estimators_.extend(estimators)

                # Validation
                if test_loader:
                    self.eval()
                    with torch.no_grad():
                        t0 = time.time()
                        val_loss = 0
                        acc = AccuracyMetric(nb_classes=self.n_outputs)
                        for _, elem in enumerate(test_loader):
                            data, target = io.split_data_target(
                                elem, self.device
                            )
                            output = _forward_eval(self.estimators_, *data)

                            loss = self._criterion(output, target)

                            val_loss += loss.item()
                            acc.update(output, target)

                        t1 = time.time() - t0

                        val_loss /= len(test_loader)
                        val_acc = acc.compute(mode="all")
                        val_acc_balanced = acc.compute(mode="class_balanced")
                        val_acc_worst = acc.compute(mode="worst_classes")

                        if val_acc_balanced > best_acc:
                            best_acc = val_acc_balanced
                            best_acc_epoch = epoch + 1
                            if save_model:
                                io.save(self, save_dir)

                        print("Epoch {} ({}) | Val loss {:.2f} | Val acc {:.2f} | Val acc (b) {:.2f} ({:.2f}) | Val acc (w) {:.2f} | Time {:.1f} seconds.".format(
                            epoch+1, best_acc_epoch, val_loss, val_acc,
                            val_acc_balanced, best_acc, val_acc_worst, t1))

                        if self.use_wandb:
                            wandb.log({"epoch": epoch+1, "val_loss": val_loss,
                                       "val_acc": val_acc,
                                       "val_acc_balanced": val_acc_balanced,
                                       "val_acc_worst": val_acc_worst,
                                       "val_epoch_time": t1})

                            wandb.run.summary["best_val_acc_balanced"] = best_acc
                            wandb.run.summary["best_val_acc_balanced_epoch"] = best_acc_epoch

                # Update the scheduler
                with warnings.catch_warnings():

                    # UserWarning raised by PyTorch is ignored because
                    # scheduler does not have a real effect on the optimizer.
                    warnings.simplefilter("ignore", UserWarning)

                    if self.use_scheduler_:
                        scheduler_.step()

        if save_model and not test_loader:
            io.save(self, save_dir)
            
    def parallel_fit_per_epoch(self, 
                                train_loader,
                                estimator,
                                cur_lr,
                                optimizer,
                                criterion,
                                idx,
                                epoch,
                                log_interval,
                                device,
                                is_classification,
                                nb_classes):
        if cur_lr:
            # Parallelization corrupts the binding between optimizer and scheduler
            set_module.update_lr(optimizer, cur_lr)

        acc = AccuracyMetric(nb_classes=nb_classes)
        train_loss = 0

        t_optim_cumul = 0
        t0 = time.time()

        for batch_idx, elem in enumerate(train_loader):

            data, target = io.split_data_target(elem, device)
            batch_size = data[0].size(0)

            t0_optim = time.time()

            optimizer.zero_grad()
            output = estimator(*data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            t_optim_cumul += time.time() - t0_optim

            train_loss += loss.item()
            acc.update(output, target)

        t1 = time.time() - t0
        time_ratio_optim = t_optim_cumul / t1

        train_loss /= len(train_loader)

        return estimator, optimizer, train_loss, acc, time_ratio_optim
    
    @classmethod
    def get_bagging_dataloaders(cls, dataloader, nb_estimators, 
                                shuffle, batch_size, num_workers, 
                                n_jobs):
        dataset = dataloader.dataset
        
        # https://github.com/pytorch/pytorch/issues/44687
        multiprocessing_context = None
        if nb_estimators is not None and nb_estimators > 1 \
            and n_jobs is not None and n_jobs > 1 \
                and num_workers is not None and num_workers > 1:
            from joblib.externals.loky.backend.context import get_context
            multiprocessing_context = get_context('loky')
        
        dataloaders = []
        for i in range(nb_estimators):
            indices = np.random.choice(len(dataset), 
                                        size=len(dataset), 
                                        replace=True)
            sub_dataset = torch.utils.data.Subset(dataset, indices)
            dataloader = torch.utils.data.DataLoader(
                sub_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                worker_init_fn=worker_init_fn,
                multiprocessing_context=multiprocessing_context
            )
            dataloaders.append(dataloader)
        return dataloaders


def worker_init_fn(worker_id):
    """
    useful for PyTorch version < 1.9
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



































































@torchensemble_model_doc(
    """Implementation on the BaggingRegressor.""", "model"
)
class BaggingRegressor(BaseRegressor):
    @torchensemble_model_doc(
        """Implementation on the data forwarding in BaggingRegressor.""",
        "regressor_forward",
    )
    def forward(self, *x):
        # Average over predictions from all base estimators.
        outputs = [estimator(*x) for estimator in self.estimators_]
        pred = op.average(outputs)

        return pred

    @torchensemble_model_doc(
        """Set the attributes on optimizer for BaggingRegressor.""",
        "set_optimizer",
    )
    def set_optimizer(self, optimizer_name, **kwargs):
        super().set_optimizer(optimizer_name, **kwargs)

    @torchensemble_model_doc(
        """Set the attributes on scheduler for BaggingRegressor.""",
        "set_scheduler",
    )
    def set_scheduler(self, scheduler_name, **kwargs):
        super().set_scheduler(scheduler_name, **kwargs)

    @torchensemble_model_doc(
        """Set the training criterion for BaggingRegressor.""",
        "set_criterion",
    )
    def set_criterion(self, criterion):
        super().set_criterion(criterion)

    @torchensemble_model_doc(
        """Implementation on the training stage of BaggingRegressor.""", "fit"
    )
    def fit(
        self,
        train_loader,
        epochs=100,
        log_interval=100,
        test_loader=None,
        save_model=True,
        save_dir=None,
    ):

        self._validate_parameters(epochs, log_interval)
        self.n_outputs = self._decide_n_outputs(train_loader)

        # Instantiate a pool of base estimators, optimizers, and schedulers.
        estimators = []
        for _ in range(self.n_estimators):
            estimators.append(self._make_estimator())

        optimizers = []
        for i in range(self.n_estimators):
            optimizers.append(
                set_module.set_optimizer(
                    estimators[i], self.optimizer_name, **self.optimizer_args
                )
            )

        if self.use_scheduler_:
            scheduler_ = set_module.set_scheduler(
                optimizers[0], self.scheduler_name, **self.scheduler_args
            )

        # Check the training criterion
        if not hasattr(self, "_criterion"):
            self._criterion = nn.MSELoss()

        # Utils
        best_loss = float("inf")

        # Internal helper function on pesudo forward
        def _forward(estimators, *x):
            outputs = [estimator(*x) for estimator in estimators]
            pred = op.average(outputs)

            return pred

        # Maintain a pool of workers
        with Parallel(n_jobs=self.n_jobs) as parallel:

            # Training loop
            for epoch in range(epochs):
                self.train()

                if self.use_scheduler_:
                    cur_lr = scheduler_.get_last_lr()[0]
                else:
                    cur_lr = None

                if self.n_jobs and self.n_jobs > 1:
                    msg = "Parallelization on the training epoch: {}"
                    print(msg.format(epoch + 1))

                rets = parallel(
                    delayed(_parallel_fit_per_epoch)(
                        train_loader,
                        estimator,
                        cur_lr,
                        optimizer,
                        self._criterion,
                        idx,
                        epoch,
                        log_interval,
                        self.device,
                        False,
                    )
                    for idx, (estimator, optimizer) in enumerate(
                        zip(estimators, optimizers)
                    )
                )

                estimators, optimizers = [], []
                for estimator, optimizer in rets:
                    estimators.append(estimator)
                    optimizers.append(optimizer)

                # Validation
                if test_loader:
                    self.eval()
                    with torch.no_grad():
                        val_loss = 0.0
                        for _, elem in enumerate(test_loader):
                            data, target = io.split_data_target(
                                elem, self.device
                            )
                            output = _forward(estimators, *data)
                            val_loss += self._criterion(output, target)
                        val_loss /= len(test_loader)

                        if val_loss < best_loss:
                            best_loss = val_loss
                            self.estimators_ = nn.ModuleList()
                            self.estimators_.extend(estimators)
                            if save_model:
                                io.save(self, save_dir)

                        msg = (
                            "Epoch: {:03d} | Validation Loss:"
                            " {:.5f} | Historical Best: {:.5f}"
                        )
                        

                # Update the scheduler
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)

                    if self.use_scheduler_:
                        scheduler_.step()

        self.estimators_ = nn.ModuleList()
        self.estimators_.extend(estimators)
        if save_model and not test_loader:
            io.save(self, save_dir)

    @torchensemble_model_doc(item="regressor_evaluate")
    def evaluate(self, test_loader):
        return super().evaluate(test_loader)

    @torchensemble_model_doc(item="predict")
    def predict(self, *x):
        return super().predict(*x)
