"""
  Snapshot ensemble generates many base estimators by enforcing a base
  estimator to converge to its local minima many times and save the
  model parameters at that point as a snapshot. The final prediction takes
  the average over predictions from all snapshot models.

  Reference:
      G. Huang, Y.-X. Li, G. Pleiss et al., Snapshot Ensemble: Train 1, and
      M for free, ICLR, 2017.
"""

import time
import math
import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

import wandb

from ._base import BaseModule, BaseClassifier, BaseRegressor
from ._base import torchensemble_model_doc
from .utils import io
from .utils import set_module
from .utils import operator as op

from .utils.hz import *

__all__ = ["SnapshotEnsembleClassifier", "SnapshotEnsembleRegressor"]


__fit_doc = """
    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        A :mod:`DataLoader` container that contains the training data.
    lr_clip : list or tuple, default=None
        Specify the accepted range of learning rate. When the learning rate
        determined by the scheduler is out of this range, it will be clipped.

        - The first element should be the lower bound of learning rate.
        - The second element should be the upper bound of learning rate.
    epochs : int, default=100
        The number of training epochs.
    log_interval : int, default=100
        The number of batches to wait before logging the training status.
    test_loader : torch.utils.data.DataLoader, default=None
        A :mod:`DataLoader` container that contains the evaluating data.

        - If ``None``, no validation is conducted after each snapshot
          being generated.
        - If not ``None``, the ensemble will be evaluated on this
          dataloader after each snapshot being generated.
    save_model : bool, default=True
        Specify whether to save the model parameters.

        - If test_loader is ``None``, the ensemble with
          ``n_estimators`` base estimators will be saved.
        - If test_loader is not ``None``, the ensemble with the best
          validation performance will be saved.
    save_dir : string, default=None
        Specify where to save the model parameters.

        - If ``None``, the model will be saved in the current directory.
        - If not ``None``, the model will be saved in the specified
          directory: ``save_dir``.
"""


def _snapshot_ensemble_model_doc(header, item="fit"):
    """
    Decorator on obtaining documentation for different snapshot ensemble
    models.
    """

    def get_doc(item):
        """Return selected item"""
        __doc = {"fit": __fit_doc}
        return __doc[item]

    def adddoc(cls):
        doc = [header + "\n\n"]
        doc.extend(get_doc(item))
        cls.__doc__ = "".join(doc)
        return cls

    return adddoc


class _BaseSnapshotEnsemble(BaseModule):
    def __init__(
        self, estimators, device, nb_classes, use_wandb
    ):
        super().__init__(estimators=estimators,
                         device=device,
                         use_wandb=use_wandb,
                         nb_classes=nb_classes,
                         n_jobs=None)
        
        for idx, estimator in enumerate(self.estimators_):
            if idx > 0:
                estimator.to("cpu")
            else:
                estimator.to(self.device)

    def _validate_parameters(self, lr_clip, epochs, log_interval):
        """Validate hyper-parameters on training the ensemble."""

        if lr_clip:
            if not (isinstance(lr_clip, list) or isinstance(lr_clip, tuple)):
                msg = "lr_clip should be a list or tuple with two elements."
                print(msg)
                raise ValueError(msg)

            if len(lr_clip) != 2:
                msg = (
                    "lr_clip should only have two elements, one for lower"
                    " bound, and another for upper bound."
                )
                print(msg)
                raise ValueError(msg)

            if not lr_clip[0] < lr_clip[1]:
                msg = (
                    "The first element = {} should be smaller than the"
                    " second element = {} in lr_clip."
                )
                print(msg.format(lr_clip[0], lr_clip[1]))
                raise ValueError(msg.format(lr_clip[0], lr_clip[1]))

        if not epochs > 0:
            msg = (
                "The number of training epochs = {} should be strictly"
                " positive."
            )
            print(msg.format(epochs))
            raise ValueError(msg.format(epochs))

        if not log_interval > 0:
            msg = (
                "The number of batches to wait before printting the"
                " training status should be strictly positive, but got {}"
                " instead."
            )
            print(msg.format(log_interval))
            raise ValueError(msg.format(log_interval))

        

    def _forward(self, *x):
        """
        Implementation on the internal data forwarding in snapshot ensemble.
        """
        # Average
        results = [estimator(*x) for estimator in self.estimators_]
        output = op.average(results)

        return output

    def _clip_lr(self, optimizer, lr_clip):
        """Clip the learning rate of the optimizer according to `lr_clip`."""
        if not lr_clip:
            return optimizer

        for param_group in optimizer.param_groups:
            if param_group["lr"] < lr_clip[0]:
                param_group["lr"] = lr_clip[0]
            if param_group["lr"] > lr_clip[1]:
                param_group["lr"] = lr_clip[1]

        return optimizer

    def _set_scheduler(self, optimizer, n_iters):
        """
        Set the learning rate scheduler for snapshot ensemble.
        Please refer to the equation (2) in original paper for details.
        """
        T_M = math.ceil(n_iters / self.n_estimators)
        lr_lambda = lambda iteration: 0.5 * (  # noqa: E731
            torch.cos(torch.tensor(math.pi * (iteration % T_M) / T_M)) + 1
        )
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return scheduler

    def set_scheduler(self, scheduler_name, **kwargs):
        msg = (
            "The learning rate scheduler for Snapshot Ensemble will be"
            " automatically set. Calling this function has no effect on"
            " the training stage of Snapshot Ensemble."
        )
        warnings.warn(msg, RuntimeWarning)


@torchensemble_model_doc(
    """Implementation on the SnapshotEnsembleClassifier.""", "seq_model"
)
class SnapshotEnsembleClassifier(_BaseSnapshotEnsemble, BaseClassifier):
    def __init__(self, estimators, device, nb_classes, use_wandb):
        super().__init__(estimators, device, nb_classes, use_wandb)
    
    
    @torchensemble_model_doc(
        """Implementation on the data forwarding in SnapshotEnsembleClassifier.""",  # noqa: E501
        "classifier_forward",
    )
    def forward(self, *x):
        proba = self._forward(*x)

        return F.softmax(proba, dim=1)

    @torchensemble_model_doc(
        """Set the attributes on optimizer for SnapshotEnsembleClassifier.""",
        "set_optimizer",
    )
    def set_optimizer(self, optimizer_name, **kwargs):
        super().set_optimizer(optimizer_name, **kwargs)

    @torchensemble_model_doc(
        """Set the training criterion for SnapshotEnsembleClassifier.""",
        "set_criterion",
    )
    def set_criterion(self, criterion):
        super().set_criterion(criterion)

    @_snapshot_ensemble_model_doc(
        """Implementation on the training stage of SnapshotEnsembleClassifier.""",  # noqa: E501
        "fit",
    )
    def fit(
        self,
        train_loader,
        lr_clip=None,
        epochs=100,
        log_interval=100,
        test_loader=None,
        save_model=True,
        save_dir=None,
    ):
        self._validate_parameters(lr_clip, epochs, log_interval)
        
        estimators = self.estimators_
        
        estimator = estimators[0]
        estimator.to(self.device)
        estimator_idx = 1
        
        self.estimators_ = nn.ModuleList()
        self.estimators_.append(estimator)

        # Set the optimizer and scheduler
        optimizer = set_module.set_optimizer(
            estimator, self.optimizer_name, **self.optimizer_args
        )

        scheduler = self._set_scheduler(optimizer, epochs * len(train_loader))

        # Check the training criterion
        if not hasattr(self, "_criterion"):
            self._criterion = nn.CrossEntropyLoss()

        # Utils
        best_acc = - 1
        best_acc_epoch = - 1
        counter = 0  # a counter on generating snapshots
        total_iters = 0
        n_iters_per_estimator = epochs * len(train_loader) // len(estimators)
        
        # Training loop
        estimator.train()
        for epoch in range(epochs):
            t0 = time.time()
            t_optim_cumul = 0
            acc = AccuracyMetric(nb_classes=self.n_outputs)
            train_loss = 0
            
            for batch_idx, elem in enumerate(train_loader):

                data, target = io.split_data_target(elem, self.device)
                batch_size = data[0].size(0)

                # Clip the learning rate
                optimizer = self._clip_lr(optimizer, lr_clip)
                
                t0_optim = time.time()

                optimizer.zero_grad()
                output = estimator(*data)
                loss = self._criterion(output, target)
                loss.backward()
                optimizer.step()
                
                t_optim_cumul += time.time() - t0_optim
                train_loss += loss.item()
                acc.update(output, target)
                
                if self.use_wandb:
                    wandb.log({
                        "epoch": epoch+float(batch_idx+1)/len(train_loader),
                        "real_time_learning_rate": optimizer.param_groups[0]["lr"]})

                # Snapshot ensemble updates the learning rate per iteration
                # instead of per epoch.
                scheduler.step()
                counter += 1
                total_iters += 1
                
                if counter % n_iters_per_estimator == 0:
                    # Generate and save the snapshot
                    if estimator_idx < len(estimators):
                        snapshot = estimators[- estimator_idx]
                        estimator_idx += 1
                        snapshot.to(self.device)
                        snapshot.load_state_dict(estimator.state_dict())
                        self.estimators_.append(snapshot)
                    
                        msg = "Save the snapshot model with index: {}"
                        print(msg.format(len(self.estimators_) - 1 + 1))
            
            t1 = time.time() - t0
            time_ratio_optim = t_optim_cumul / t1
            train_loss /= len(train_loader)
            train_acc = acc.compute(mode="all")
            train_acc_balanced = acc.compute(mode="class_balanced")
            train_acc_worst = acc.compute(mode="worst_classes")
            
            print("Epoch {} | LR: {} | Train loss {:.2f} | Train acc {:.2f} | Train acc (b) {:.2f} | Train acc (w) {:.2f} | Time {:.1f} seconds. Optimization time ratio: {:.2f}.".format(
                epoch+1, optimizer.param_groups[0]["lr"], train_loss, 
                train_acc, train_acc_balanced,
                train_acc_worst, t1, time_ratio_optim))
            if self.use_wandb:
                wandb.log({"epoch": epoch+1, "train_loss": train_loss,
                           "train_acc": train_acc, "train_acc_balanced": train_acc_balanced,
                           "train_acc_worst": train_acc_worst,
                            "train_epoch_time": t1, 
                            "time_ratio_optim": time_ratio_optim, 
                            "real_time_learning_rate": optimizer.param_groups[0]["lr"]})
                    
            # Validation after each training epoch
            if test_loader:
                self.eval()
                with torch.no_grad():
                    
                    t0 = time.time()
                    acc = AccuracyMetric(nb_classes=self.n_outputs)
                    val_loss = 0
                    
                    for _, elem in enumerate(test_loader):
                        data, target = io.split_data_target(elem, self.device)
                        output = self.forward(*data)
                        
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
                            
                    print("Epoch {} ({}) | {} estimators | Val loss {:.2f} | Val acc {:.2f} | Val acc (b) {:.2f} ({:.2f}) | Val acc (w) {:.2f} | Time {:.1f} seconds.".format(
                        epoch+1, best_acc_epoch, len(self.estimators_), 
                        val_loss,
                        val_acc, val_acc_balanced, best_acc, 
                        val_acc_worst, t1))
                    if self.use_wandb:
                        wandb.log({"epoch": epoch+1, "val_loss": val_loss,
                                   "val_acc": val_acc, "val_acc_balanced": val_acc_balanced,
                                   "val_acc_worst": val_acc_worst,
                                   "val_epoch_time": t1, 
                                   "nb_existing_estimators": len(self.estimators_)})
                        wandb.run.summary["best_val_acc_balanced"] = best_acc
                        wandb.run.summary["best_val_acc_balanced_epoch"] = best_acc_epoch

                    
        
        if save_model and not test_loader:
            io.save(self, save_dir)
    

    @torchensemble_model_doc(item="classifier_evaluate")
    def evaluate(self, test_loader, return_loss=False):
        return super().evaluate(test_loader, return_loss)

    @torchensemble_model_doc(item="predict")
    def predict(self, *x):
        return super().predict(*x)


@torchensemble_model_doc(
    """Implementation on the SnapshotEnsembleRegressor.""", "seq_model"
)
class SnapshotEnsembleRegressor(_BaseSnapshotEnsemble, BaseRegressor):
    @torchensemble_model_doc(
        """Implementation on the data forwarding in SnapshotEnsembleRegressor.""",  # noqa: E501
        "regressor_forward",
    )
    def forward(self, *x):
        pred = self._forward(*x)
        return pred

    @torchensemble_model_doc(
        """Set the attributes on optimizer for SnapshotEnsembleRegressor.""",
        "set_optimizer",
    )
    def set_optimizer(self, optimizer_name, **kwargs):
        super().set_optimizer(optimizer_name, **kwargs)

    @torchensemble_model_doc(
        """Set the training criterion for SnapshotEnsembleRegressor.""",
        "set_criterion",
    )
    def set_criterion(self, criterion):
        super().set_criterion(criterion)

    @_snapshot_ensemble_model_doc(
        """Implementation on the training stage of SnapshotEnsembleRegressor.""",  # noqa: E501
        "fit",
    )
    def fit(
        self,
        train_loader,
        lr_clip=None,
        epochs=100,
        log_interval=100,
        test_loader=None,
        save_model=True,
        save_dir=None,
    ):
        self._validate_parameters(lr_clip, epochs, log_interval)
        self.n_outputs = self._decide_n_outputs(train_loader)

        estimator = self._make_estimator()

        # Set the optimizer and scheduler
        optimizer = set_module.set_optimizer(
            estimator, self.optimizer_name, **self.optimizer_args
        )

        scheduler = self._set_scheduler(optimizer, epochs * len(train_loader))

        # Check the training criterion
        if not hasattr(self, "_criterion"):
            self._criterion = nn.MSELoss()

        # Utils
        best_loss = float("inf")
        counter = 0  # a counter on generating snapshots
        total_iters = 0
        n_iters_per_estimator = epochs * len(train_loader) // self.n_estimators

        # Training loop
        estimator.train()
        for epoch in range(epochs):
            for batch_idx, elem in enumerate(train_loader):

                data, target = io.split_data_target(elem, self.device)

                # Clip the learning rate
                optimizer = self._clip_lr(optimizer, lr_clip)

                optimizer.zero_grad()
                output = estimator(*data)
                loss = self._criterion(output, target)
                loss.backward()
                optimizer.step()

                # Print training status
                if batch_idx % log_interval == 0:
                    with torch.no_grad():
                        msg = (
                            "lr: {:.5f} | Epoch: {:03d} | Batch: {:03d}"
                            " | Loss: {:.5f}"
                        )
                        print(
                            msg.format(
                                optimizer.param_groups[0]["lr"],
                                epoch,
                                batch_idx,
                                loss,
                            )
                        )
                        

                # Snapshot ensemble updates the learning rate per iteration
                # instead of per epoch.
                scheduler.step()
                counter += 1
                total_iters += 1

            if counter % n_iters_per_estimator == 0:
                # Generate and save the snapshot
                snapshot = self._make_estimator()
                snapshot.load_state_dict(estimator.state_dict())
                self.estimators_.append(snapshot)

                msg = "Save the snapshot model with index: {}"
                print(msg.format(len(self.estimators_) - 1))

            # Validation after each snapshot model being generated
            if test_loader and counter % n_iters_per_estimator == 0:
                self.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    for _, elem in enumerate(test_loader):
                        data, target = io.split_data_target(elem, self.device)
                        output = self.forward(*data)
                        val_loss += self._criterion(output, target)
                    val_loss /= len(test_loader)

                    if val_loss < best_loss:
                        best_loss = val_loss
                        if save_model:
                            io.save(self, save_dir)

                    msg = (
                        "n_estimators: {} | Validation Loss: {:.5f} |"
                        " Historical Best: {:.5f}"
                    )
                    print(
                        msg.format(len(self.estimators_), val_loss, best_loss)
                    )
                    

        if save_model and not test_loader:
            io.save(self, save_dir)

    @torchensemble_model_doc(item="regressor_evaluate")
    def evaluate(self, test_loader):
        return super().evaluate(test_loader)

    @torchensemble_model_doc(item="predict")
    def predict(self, *x):
        return super().predict(*x)
