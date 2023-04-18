"""
  Motivated by geometric insights on the loss surface of deep neural networks,
  Fast Geometirc Ensembling (FGE) is an efficient ensemble that uses a
  customized learning rate scheduler to generate base estimators, similar to
  snapshot ensemble.

  Reference:
      T. Garipov, P. Izmailov, D. Podoprikhin et al., Loss Surfaces, Mode
      Connectivity, and Fast Ensembling of DNNs, NeurIPS, 2018.
"""

import time
import copy
import torch
import warnings
import math
import torch.nn as nn
import torch.nn.functional as F

import wandb

from ._base import BaseModule, BaseClassifier, BaseRegressor
from ._base import torchensemble_model_doc
from .utils import io
from .utils import set_module
from .utils import operator as op

from .utils.hz import * 

__all__ = ["FastGeometricClassifier", "FastGeometricRegressor", 
           "FastGeometricClassifierM", "FastGeometricClassifierMv2"]


__fit_doc = """
    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        A :mod:`DataLoader` container that contains the training data.
    cycle : int, default=4
        The number of cycles used to build each base estimator in the ensemble.
    lr_1 : float, default=5e-2
        ``alpha_1`` in original paper used to adjust the learning rate, also
        serves as the initial learning rate of the internal optimizer.
    lr_2 : float, default=1e-4
        ``alpha_2`` in original paper used to adjust the learning rate, also
        serves as the smallest learning rate of the internal optimizer.
    epochs : int, default=100
        The number of training epochs used to fit the dummy base estimator.
    log_interval : int, default=100
        The number of batches to wait before logging the training status.
    test_loader : torch.utils.data.DataLoader, default=None
        A :mod:`DataLoader` container that contains the evaluating data.

        - If ``None``, no validation is conducted after each real base
          estimator being generated.
        - If not ``None``, the ensemble will be evaluated on this
          dataloader after each base estimator being generated.
    save_model : bool, default=True
        Specify whether to save the model parameters.

        - If test_loader is ``None``, the ensemble fully trained will be
          saved.
        - If test_loader is not ``None``, the ensemble with the best
          validation performance will be saved.
    save_dir : string, default=None
        Specify where to save the model parameters.

        - If ``None``, the model will be saved in the current directory.
        - If not ``None``, the model will be saved in the specified
          directory: ``save_dir``.
"""


def _fast_geometric_model_doc(header, item="fit"):
    """
    Decorator on obtaining documentation for different fast geometric models.
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


class _BaseFastGeometric(BaseModule):
    def __init__(
        self, estimators, device, nb_classes, use_wandb
    ):
        super().__init__(estimators=estimators, 
                         device=device, 
                         n_jobs=None, 
                         use_wandb=use_wandb,
                         nb_classes=nb_classes)
        
        for idx, estimator in enumerate(self.estimators_):
            estimator.to("cpu")


    def _forward(self, *x):
        """
        Implementation on the internal data forwarding in fast geometric
        ensemble.
        """
        # Average
        results = [estimator(*x) for estimator in self.estimators_]
        output = op.average(results)

        return output

    def _adjust_lr(
        self, optimizer, epoch, i, n_iters, cycle, alpha_1, alpha_2
    ):
        """
        Set the internal learning rate scheduler for fast geometric ensemble.
        Please refer to the original paper for details.
        """

        def scheduler(i):
            t = ((epoch % cycle) + i) / cycle
            if t < 0.5:
                return alpha_1 * (1.0 - 2.0 * t) + alpha_2 * 2.0 * t
            else:
                return alpha_1 * (2.0 * t - 1.0) + alpha_2 * (2.0 - 2.0 * t)

        lr = scheduler(i / n_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        return lr

    @torchensemble_model_doc(
        """Set the attributes on optimizer for Fast Geometric Ensemble.""",
        "set_optimizer",
    )
    def set_optimizer(self, optimizer_name, **kwargs):
        self.optimizer_name = optimizer_name
        self.optimizer_args = kwargs

    @torchensemble_model_doc(
        """Set the attributes on scheduler for Fast Geometric Ensemble.""",
        "set_scheduler",
    )
    def set_scheduler(self, scheduler_name, **kwargs):
        msg = (
            "The learning rate scheduler for fast geometirc ensemble will"
            " only be used in the first stage on building the dummy base"
            " estimator."
        )
        warnings.warn(msg, UserWarning)

        self.scheduler_name = scheduler_name
        self.scheduler_args = kwargs
        self.use_scheduler_ = True


@torchensemble_model_doc(
    """Implementation on the FastGeometricClassifier.""", "seq_model"
)
class FastGeometricClassifier(_BaseFastGeometric, BaseClassifier):
    def __init__(self, estimators, device, nb_classes, use_wandb):
        super().__init__(estimators, device, nb_classes, use_wandb)
    
    @torchensemble_model_doc(
        """Implementation on the data forwarding in FastGeometricClassifier.""",  # noqa: E501
        "classifier_forward",
    )
    def forward(self, *x):
        proba = self._forward(*x)

        return F.softmax(proba, dim=1)

    @torchensemble_model_doc(
        (
            """Set the attributes on optimizer for FastGeometricClassifier. """
            + """Notice that keyword arguments specified here will also be """
            + """used in the ensembling stage except the learning rate.."""
        ),
        "set_optimizer",
    )
    def set_optimizer(self, optimizer_name, **kwargs):
        super().set_optimizer(optimizer_name=optimizer_name, **kwargs)

    @torchensemble_model_doc(
        (
            """Set the attributes on scheduler for FastGeometricClassifier. """
            + """Notice that this scheduler will only be used in the stage on """  # noqa: E501
            + """fitting the dummy base estimator."""
        ),
        "set_scheduler",
    )
    def set_scheduler(self, scheduler_name, **kwargs):
        super().set_scheduler(scheduler_name=scheduler_name, **kwargs)

    @torchensemble_model_doc(
        """Set the training criterion for FastGeometricClassifier.""",
        "set_criterion",
    )
    def set_criterion(self, criterion):
        super().set_criterion(criterion)

    @_fast_geometric_model_doc(
        """Implementation on the training stage of FastGeometricClassifier.""",  # noqa: E501
        "fit",
    )
    def fit(
        self,
        train_loader,
        cycle=4,
        lr_1=5e-2,
        lr_2=1e-4,
        epochs=100,
        log_interval=100,
        test_loader=None,
        save_model=True,
        save_dir=None,
    ):
        self._validate_parameters(epochs, log_interval)
        

        # ====================================================================
        #                Train the dummy estimator (estimator_)
        # ====================================================================

        estimator_ = copy.deepcopy(self.estimators_[0])
        estimator_.to(self.device)
        
        self.n_estimators = len(self.estimators_)
        self.estimators_ = nn.ModuleList()

        # Set the optimizer and scheduler
        optimizer = set_module.set_optimizer(
            estimator_, self.optimizer_name, **self.optimizer_args
        )

        if self.use_scheduler_:
            scheduler = set_module.set_scheduler(
                optimizer, self.scheduler_name, **self.scheduler_args
            )

        # Check the training criterion
        if not hasattr(self, "_criterion"):
            self._criterion = nn.CrossEntropyLoss()

        # Utils
        total_iters = 0

        dummy_epochs = int(epochs - cycle * (self.n_estimators - 1) - cycle // 2)
        
        if self.use_wandb:
            wandb.run.summary["dummy_epochs"] = dummy_epochs
        
        assert dummy_epochs > 0

        for epoch in range(dummy_epochs):

            # Training
            estimator_.train()
            
            t0 = time.time()
            t_optim_cumul = 0
            acc = AccuracyMetric(nb_classes=self.n_outputs)
            train_loss = 0
            
            for batch_idx, elem in enumerate(train_loader):

                data, target = io.split_data_target(elem, self.device)
                batch_size = data[0].size(0)
                
                t0_optim = time.time()

                optimizer.zero_grad()
                output = estimator_(*data)
                loss = self._criterion(output, target)
                loss.backward()
                optimizer.step()
                
                t_optim_cumul += time.time() - t0_optim
                train_loss += loss.item()
                acc.update(output, target)
                        
                total_iters += 1

            t1 = time.time() - t0
            time_ratio_optim = t_optim_cumul / t1
            train_loss /= len(train_loader)
            train_acc = acc.compute(mode="all")
            train_acc_balanced = acc.compute(mode="class_balanced")
            train_acc_worst = acc.compute(mode="worst_classes")

            print("Epoch {} (dummy) | LR: {} | Train loss {:.2f} | Train acc {:.2f} | Train acc (b) {:.2f} | Train acc (w) {:.2f} | Time {:.1f} seconds. Optimization time ratio: {:.2f}.".format(
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

            if self.use_scheduler_:
                scheduler.step()
                
            test_dummy = True
            if test_dummy:
                t0 = time.time()

                acc = AccuracyMetric(nb_classes=self.n_outputs)

                estimator_.eval()
                with torch.no_grad():
                    for _, elem in enumerate(test_loader):
                        data, target = io.split_data_target(elem, self.device)

                        output = estimator_(*data)

                        acc.update(output, target)

                val_acc = acc.compute(mode="all")
                val_acc_balanced = acc.compute(mode="class_balanced")
                val_acc_worst = acc.compute(mode="worst_classes")

                t1 = time.time() - t0

                print("Epoch {} (dummy) | Val acc {:.2f} | Val acc (b) {:.2f} | Val acc (w) {:.2f} | Time {:.1f} seconds.".format(
                    epoch+1, val_acc,
                    val_acc_balanced, val_acc_worst, t1))

                if self.use_wandb:
                    wandb.log({"epoch": epoch+1, 
                            "val_acc": val_acc,
                            "val_acc_balanced": val_acc_balanced,
                            "val_acc_worst": val_acc_worst,
                            "val_epoch_time": t1})

        # ====================================================================
        #                        Generate the ensemble
        # ====================================================================
        
        # Set the internal optimizer
        estimator_.zero_grad()
        optimizer = set_module.set_optimizer(
            estimator_, self.optimizer_name, **self.optimizer_args
        )

        # Utils
        best_acc = - 1
        best_acc_epoch = - 1
        n_iters = len(train_loader)
        updated = False
        epoch = 0
        
        
        while len(self.estimators_) < self.n_estimators:

            # Training
            estimator_.train()
            
            t0 = time.time()
            t_optim_cumul = 0
            acc = AccuracyMetric(nb_classes=self.n_outputs)
            train_loss = 0
            
            for batch_idx, elem in enumerate(train_loader):

                # Update learning rate
                self._adjust_lr(
                    optimizer, epoch, batch_idx, n_iters, cycle, lr_1, lr_2
                )

                data, target = io.split_data_target(elem, self.device)
                batch_size = data[0].size(0)
                
                t0_optim = time.time()

                optimizer.zero_grad()
                output = estimator_(*data)
                loss = self._criterion(output, target)
                loss.backward()
                optimizer.step()
                
                t_optim_cumul += time.time() - t0_optim
                train_loss += loss.item()
                acc.update(output, target)
                        
                total_iters += 1
                
                if self.use_wandb:
                    wandb.log(
                        {"epoch": epoch+dummy_epochs+float(batch_idx+1)/len(train_loader), 
                         "real_time_learning_rate": optimizer.param_groups[0]["lr"]})

            t1 = time.time() - t0
            time_ratio_optim = t_optim_cumul / t1
            train_loss /= len(train_loader)
            train_acc = acc.compute(mode="all")
            train_acc_balanced = acc.compute(mode="class_balanced")
            train_acc_worst = acc.compute(mode="worst_classes")

            print("Epoch {} | LR: {} | Train loss {:.2f} | Train acc {:.2f} | Train acc (b) {:.2f} | Train acc (w) {:.2f} | Time {:.1f} seconds. Optimization time ratio: {:.2f}.".format(
                epoch+1+dummy_epochs, optimizer.param_groups[0]["lr"], train_loss,
                train_acc, train_acc_balanced,
                train_acc_worst, t1, time_ratio_optim))
            if self.use_wandb:
                wandb.log({"epoch": epoch+1+dummy_epochs, "train_loss": train_loss,
                           "train_acc": train_acc, "train_acc_balanced": train_acc_balanced,
                           "train_acc_worst": train_acc_worst,
                           "train_epoch_time": t1,
                           "time_ratio_optim": time_ratio_optim,
                           "real_time_learning_rate": optimizer.param_groups[0]["lr"]})

            # Update the ensemble
            if (epoch % cycle + 1) == cycle // 2:    
                base_estimator = copy.deepcopy(estimator_)
                base_estimator.to(self.device)
                base_estimator.load_state_dict(estimator_.state_dict()) # redundant
                
                self.estimators_.append(base_estimator)
                updated = True
                total_iters = 0

                msg = "Save the base estimator with index: {}"
                print(msg.format(len(self.estimators_) - 1))

            # Validation after each base estimator being added
            if test_loader and updated:
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
                        epoch+1+dummy_epochs, best_acc_epoch, len(self.estimators_),
                        val_loss,
                        val_acc, val_acc_balanced, best_acc,
                        val_acc_worst, t1))
                    if self.use_wandb:
                        wandb.log({"epoch": epoch+1+dummy_epochs, "val_loss": val_loss,
                                   "val_acc": val_acc, "val_acc_balanced": val_acc_balanced,
                                   "val_acc_worst": val_acc_worst,
                                   "val_epoch_time": t1,
                                   "nb_existing_estimators": len(self.estimators_)})
                        wandb.run.summary["best_val_acc_balanced"] = best_acc
                        wandb.run.summary["best_val_acc_balanced_epoch"] = best_acc_epoch

                    
                updated = False  # reset the updating flag
            epoch += 1

        if save_model and not test_loader:
            io.save(self, save_dir)

    @torchensemble_model_doc(item="classifier_evaluate")
    def evaluate(self, test_loader, return_loss=False):
        return super().evaluate(test_loader, return_loss)

    @torchensemble_model_doc(item="predict")
    def predict(self, *x):
        return super().predict(*x)


class FastGeometricClassifierM(FastGeometricClassifier):
    def __init__(self, estimators, device, nb_classes, use_wandb):
        super().__init__(estimators, device, nb_classes, use_wandb)
    
    def forward_eval(self, *x):
        stem = self.estimators_[0].stem
        intermediate_feature = stem(*x)
        results = [estimator.branch(intermediate_feature) for estimator in self.estimators_]
        proba = op.average(results)

        return F.softmax(proba, dim=1)
        
    def update_ensemble_and_test(self, 
                                 iteration_idx, 
                                 iterations_per_estimator, 
                                 estimator_,
                                 test_loader, 
                                 epoch,
                                 save_model, 
                                 save_dir,
                                 dummy_epochs, 
                                 best_acc,
                                 best_acc_epoch,
                                 n_iters,
                                 updated,
                                 total_iters,
                                 epoch_fine_grained):
        
        # Update the ensemble
        t_curr = ((iteration_idx - 1) %
                  iterations_per_estimator) / iterations_per_estimator
        t_nxt = (iteration_idx % iterations_per_estimator) / \
            iterations_per_estimator
        if t_curr == 0.5 or (t_curr <= 0.5 and t_nxt > 0.5):
            base_estimator = copy.deepcopy(estimator_)
            base_estimator.to(self.device)
            #base_estimator.load_state_dict(estimator_.state_dict())  # redundant
            
            self.estimators_.append(base_estimator)
            updated = True
            total_iters = 0

            msg = "Save the base estimator with index: {}"
            print(msg.format(len(self.estimators_) - 1))

        # Validation after each base estimator being added
        if test_loader and updated:
            self.eval()
            with torch.no_grad():

                t0 = time.time()
                acc = AccuracyMetric(nb_classes=self.n_outputs)
                val_loss = 0

                for _, elem in enumerate(test_loader):
                    data, target = io.split_data_target(
                        elem, self.device)
                    output = self.forward_eval(*data)
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

                print("Epoch {} ({}) | {} estimators | Val loss {:.2f} | Val acc {:.2f} | Val acc (b) {:.2f} ({:.2f}) | Val acc (w) {:.2f} | Time {:.1f} seconds.".format(
                    epoch_fine_grained, best_acc_epoch, len(
                        self.estimators_),
                    val_loss,
                    val_acc, val_acc_balanced, best_acc,
                    val_acc_worst, t1))
                if self.use_wandb:
                    wandb.log({"epoch": epoch_fine_grained, "val_loss": val_loss,
                               "val_acc": val_acc, "val_acc_balanced": val_acc_balanced,
                               "val_acc_worst": val_acc_worst,
                               "val_epoch_time": t1,
                               "nb_existing_estimators": len(self.estimators_)})
                    wandb.run.summary["best_val_acc_balanced"] = best_acc
                    wandb.run.summary["best_val_acc_balanced_epoch"] = best_acc_epoch

            updated = False  # reset the updating flag
            
        return iteration_idx, iterations_per_estimator, estimator_, \
            test_loader, epoch, save_model, save_dir, dummy_epochs, \
            best_acc, best_acc_epoch, n_iters, updated, total_iters
    
    def _adjust_lr(
        self, optimizer, iteration_idx, iterations_per_estimator, alpha_1, alpha_2
    ):
        """
        Set the internal learning rate scheduler for fast geometric ensemble.
        Please refer to the original paper for details.
        """

        def scheduler(iteration_idx, iterations_per_estimator):
            t = (iteration_idx % iterations_per_estimator) / iterations_per_estimator
            if t < 0.5:
                return alpha_1 * (1.0 - 2.0 * t) + alpha_2 * 2.0 * t
            else:
                return alpha_1 * (2.0 * t - 1.0) + alpha_2 * (2.0 - 2.0 * t)

        lr = scheduler(iteration_idx, iterations_per_estimator)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        return lr
    
    def get_dummy_epochs(self, epochs, train_loader):
        iterations_per_estimator = - 1
        ratio_ = 0.8
        while iterations_per_estimator < 2:
        
            dummy_epochs = int(math.floor(epochs * ratio_))

            total_iterations = (epochs - dummy_epochs) * len(train_loader)
            iterations_per_estimator = int(math.floor(
                total_iterations / self.n_estimators))
            
            ratio_ = ratio_ - 0.1
            if ratio_ == 0:
                raise Exception(
                    """There are too many branches compared 
                    to the total number of epochs, batch size and 
                    number of training examples.""")

        return dummy_epochs, iterations_per_estimator, total_iterations

    def fit(
        self,
        train_loader,
        lr_1=5e-2,
        lr_2=1e-4,
        epochs=100,
        log_interval=100,
        test_loader=None,
        save_model=True,
        save_dir=None,
    ):
        self._validate_parameters(epochs, log_interval)

        # ====================================================================
        #                Train the dummy estimator (estimator_)
        # ====================================================================

        estimator_ = copy.deepcopy(self.estimators_[0])
        estimator_.to(self.device)

        self.n_estimators = len(self.estimators_)
        self.estimators_ = nn.ModuleList()

        # Set the optimizer and scheduler
        optimizer = set_module.set_optimizer(
            estimator_, self.optimizer_name, **self.optimizer_args
        )

        if self.use_scheduler_:
            scheduler = set_module.set_scheduler(
                optimizer, self.scheduler_name, **self.scheduler_args
            )

        # Check the training criterion
        if not hasattr(self, "_criterion"):
            self._criterion = nn.CrossEntropyLoss()

        # Utils
        total_iters = 0

        # total_iterations is used if one only wants to test the test dataset 
        # at the last moment
        dummy_epochs, iterations_per_estimator, total_iterations = self.get_dummy_epochs(
            epochs, train_loader)

        if self.use_wandb:
            wandb.run.summary["dummy_epochs"] = dummy_epochs

        assert dummy_epochs > 0

        for epoch in range(dummy_epochs):

            # Training
            estimator_.train()

            t0 = time.time()
            t_optim_cumul = 0
            acc = AccuracyMetric(nb_classes=self.n_outputs)
            train_loss = 0

            for batch_idx, elem in enumerate(train_loader):

                data, target = io.split_data_target(elem, self.device)
                batch_size = data[0].size(0)

                t0_optim = time.time()

                optimizer.zero_grad()
                output = estimator_(*data)
                loss = self._criterion(output, target)
                loss.backward()
                optimizer.step()

                t_optim_cumul += time.time() - t0_optim
                train_loss += loss.item()
                acc.update(output, target)

                total_iters += 1

            t1 = time.time() - t0
            time_ratio_optim = t_optim_cumul / t1
            train_loss /= len(train_loader)
            train_acc = acc.compute(mode="all")
            train_acc_balanced = acc.compute(mode="class_balanced")
            train_acc_worst = acc.compute(mode="worst_classes")

            print("Epoch {} (dummy) | LR: {} | Train loss {:.2f} | Train acc {:.2f} | Train acc (b) {:.2f} | Train acc (w) {:.2f} | Time {:.1f} seconds. Optimization time ratio: {:.2f}.".format(
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

            if self.use_scheduler_:
                scheduler.step()

            test_dummy = True
            if test_dummy:
                t0 = time.time()

                acc = AccuracyMetric(nb_classes=self.n_outputs)

                estimator_.eval()
                with torch.no_grad():
                    for _, elem in enumerate(test_loader):
                        data, target = io.split_data_target(elem, self.device)

                        output = estimator_(*data)

                        acc.update(output, target)

                val_acc = acc.compute(mode="all")
                val_acc_balanced = acc.compute(mode="class_balanced")
                val_acc_worst = acc.compute(mode="worst_classes")

                t1 = time.time() - t0

                print("Epoch {} (dummy) | Val acc {:.2f} | Val acc (b) {:.2f} | Val acc (w) {:.2f} | Time {:.1f} seconds.".format(
                    epoch+1, val_acc,
                    val_acc_balanced, val_acc_worst, t1))

                if self.use_wandb:
                    wandb.log({"epoch": epoch+1,
                               "val_acc": val_acc,
                               "val_acc_balanced": val_acc_balanced,
                               "val_acc_worst": val_acc_worst,
                               "val_epoch_time": t1})

        # ====================================================================
        #                        Generate the ensemble
        # ====================================================================

        # Set the internal optimizer
        estimator_.zero_grad()
        optimizer = set_module.set_optimizer(
            estimator_, self.optimizer_name, **self.optimizer_args
        )

        # Utils
        best_acc = - 1
        best_acc_epoch = - 1
        n_iters = len(train_loader)
        updated = False
        epoch = 0
        
        iteration_idx = 1

        while len(self.estimators_) < self.n_estimators:

            # Training
            estimator_.train()

            t0 = time.time()
            t_optim_cumul = 0
            t_update_ensemble_and_test = 0
            acc = AccuracyMetric(nb_classes=self.n_outputs)
            train_loss = 0

            for batch_idx, elem in enumerate(train_loader):

                # Update learning rate
                self._adjust_lr(
                    optimizer, iteration_idx, iterations_per_estimator, lr_1, lr_2
                )
                
                iteration_idx = iteration_idx + 1

                data, target = io.split_data_target(elem, self.device)
                batch_size = data[0].size(0)

                t0_optim = time.time()

                optimizer.zero_grad()
                output = estimator_(*data)
                loss = self._criterion(output, target)
                loss.backward()
                optimizer.step()

                t_optim_cumul += time.time() - t0_optim
                train_loss += loss.item()
                acc.update(output, target)

                total_iters += 1
                
                epoch_fine_grained = epoch+dummy_epochs+float(batch_idx+1)/len(train_loader)

                if self.use_wandb:
                    wandb.log(
                        {"epoch": epoch_fine_grained,
                         "real_time_learning_rate": optimizer.param_groups[0]["lr"]})
                
                t0_update_ensemble_and_test = time.time()
                
                iteration_idx, iterations_per_estimator, estimator_, \
                    test_loader, epoch, save_model, save_dir, dummy_epochs, \
                        best_acc, best_acc_epoch, n_iters, updated, total_iters = \
                            self.update_ensemble_and_test(iteration_idx, 
                                                            iterations_per_estimator, 
                                                            estimator_,
                                                            test_loader, 
                                                            epoch,
                                                            save_model, 
                                                            save_dir,
                                                            dummy_epochs, 
                                                            best_acc,
                                                            best_acc_epoch,
                                                            n_iters,
                                                            updated,
                                                            total_iters,
                                                            epoch_fine_grained) 
                t_update_ensemble_and_test += time.time() - t0_update_ensemble_and_test
            
            t1 = time.time() - t0 - t_update_ensemble_and_test 
            time_ratio_optim = t_optim_cumul / t1
            train_loss /= len(train_loader)
            train_acc = acc.compute(mode="all")
            train_acc_balanced = acc.compute(mode="class_balanced")
            train_acc_worst = acc.compute(mode="worst_classes")

            print("Epoch {} | LR: {} | Train loss {:.2f} | Train acc {:.2f} | Train acc (b) {:.2f} | Train acc (w) {:.2f} | Time {:.1f} seconds. Optimization time ratio: {:.2f}.".format(
                epoch+1 +
                dummy_epochs, optimizer.param_groups[0]["lr"], train_loss,
                train_acc, train_acc_balanced,
                train_acc_worst, t1, time_ratio_optim))
            if self.use_wandb:
                wandb.log({"epoch": epoch+1+dummy_epochs, "train_loss": train_loss,
                           "train_acc": train_acc, "train_acc_balanced": train_acc_balanced,
                           "train_acc_worst": train_acc_worst,
                           "train_epoch_time": t1,
                           "time_ratio_optim": time_ratio_optim,
                           "real_time_learning_rate": optimizer.param_groups[0]["lr"]})

            
            epoch += 1

        if save_model and not test_loader:
            io.save(self, save_dir)


class FastGeometricClassifierMv2(FastGeometricClassifier):
    def __init__(self, estimators, device, nb_classes, use_wandb):
        super().__init__(estimators, device, nb_classes, use_wandb)

    def forward_eval(self, *x):
        stem = self.estimators_[0].stem
        intermediate_feature = stem(*x)
        results = [estimator.branch(intermediate_feature)
                   for estimator in self.estimators_]
        proba = op.average(results)

        return F.softmax(proba, dim=1)

    def update_ensemble_and_test(self,
                                 iteration_idx,
                                 iterations_per_estimator,
                                 estimator_,
                                 test_loader,
                                 epoch,
                                 save_model,
                                 save_dir,
                                 dummy_epochs,
                                 best_acc,
                                 best_acc_epoch,
                                 n_iters,
                                 updated,
                                 total_iters):

        # Update the ensemble
        t_curr = ((iteration_idx - 1) %
                  iterations_per_estimator) / iterations_per_estimator
        t_nxt = (iteration_idx % iterations_per_estimator) / \
            iterations_per_estimator
        if t_curr == 0.5 or (t_curr <= 0.5 and t_nxt > 0.5):
            base_estimator = copy.deepcopy(estimator_)
            #base_estimator.to(self.device)
            

            self.estimators_.append(base_estimator)
            updated = True
            total_iters = 0

            msg = "Save the base estimator with index: {}"
            print(msg.format(len(self.estimators_) - 1))

        return iteration_idx, iterations_per_estimator, estimator_, \
            test_loader, epoch, save_model, save_dir, dummy_epochs, \
            best_acc, best_acc_epoch, n_iters, updated, total_iters

    def _adjust_lr(
        self, optimizer, iteration_idx, iterations_per_estimator, alpha_1, alpha_2
    ):
        """
        Set the internal learning rate scheduler for fast geometric ensemble.
        Please refer to the original paper for details.
        """

        def scheduler(iteration_idx, iterations_per_estimator):
            t = (iteration_idx % iterations_per_estimator) / \
                iterations_per_estimator
            if t < 0.5:
                return alpha_1 * (1.0 - 2.0 * t) + alpha_2 * 2.0 * t
            else:
                return alpha_1 * (2.0 * t - 1.0) + alpha_2 * (2.0 - 2.0 * t)

        lr = scheduler(iteration_idx, iterations_per_estimator)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        return lr

    def get_dummy_epochs(self, epochs, train_loader):
        iterations_per_estimator = - 1
        ratio_ = 0.8
        while iterations_per_estimator < 2:

            dummy_epochs = int(math.floor(epochs * ratio_))

            total_iterations = (epochs - dummy_epochs) * len(train_loader)
            iterations_per_estimator = int(math.floor(
                total_iterations / self.n_estimators))

            ratio_ = ratio_ - 0.1
            if ratio_ == 0:
                raise Exception(
                    """There are too many branches compared 
                    to the total number of epochs, batch size and 
                    number of training examples.""")

        return dummy_epochs, iterations_per_estimator, total_iterations

    def fit(
        self,
        train_loader,
        lr_1=5e-2,
        lr_2=1e-4,
        epochs=100,
        log_interval=100,
        test_loader=None,
        save_model=True,
        save_dir=None,
    ):
        self._validate_parameters(epochs, log_interval)

        # ====================================================================
        #                Train the dummy estimator (estimator_)
        # ====================================================================

        estimator_ = copy.deepcopy(self.estimators_[0])
        estimator_.to(self.device)

        self.n_estimators = len(self.estimators_)
        self.estimators_ = nn.ModuleList()

        # Set the optimizer and scheduler
        optimizer = set_module.set_optimizer(
            estimator_, self.optimizer_name, **self.optimizer_args
        )

        if self.use_scheduler_:
            scheduler = set_module.set_scheduler(
                optimizer, self.scheduler_name, **self.scheduler_args
            )

        # Check the training criterion
        if not hasattr(self, "_criterion"):
            self._criterion = nn.CrossEntropyLoss()

        # Utils
        total_iters = 0

        # total_iterations is used if one only wants to test the test dataset
        # at the last moment
        dummy_epochs, iterations_per_estimator, total_iterations = self.get_dummy_epochs(
            epochs, train_loader)

        if self.use_wandb:
            wandb.run.summary["dummy_epochs"] = dummy_epochs

        assert dummy_epochs > 0

        for epoch in range(dummy_epochs):

            # Training
            estimator_.train()

            t0 = time.time()
            t_optim_cumul = 0
            acc = QuickAccuracyMetric()
            train_loss = 0

            for elem in train_loader:

                data, target = io.split_data_target(elem, self.device)
                #batch_size = data[0].size(0)

                t0_optim = time.time()

                optimizer.zero_grad()
                output = estimator_(*data)
                loss = self._criterion(output, target)
                loss.backward()
                optimizer.step()

                t_optim_cumul += time.time() - t0_optim
                train_loss += loss.item()
                acc.update(output, target)

                total_iters += 1

            t1 = time.time() - t0
            time_ratio_optim = t_optim_cumul / t1
            train_loss /= len(train_loader)
            train_acc = acc.compute()
            

            print("Epoch {} | Train loss {:.2f} | Train acc {:.2f} | Time {:.1f} ({:.1f}) seconds. Optimization time ratio: {:.2f}.".format(
                epoch, train_loss, train_acc, t1, t_optim_cumul, time_ratio_optim))
            if self.use_wandb:
                wandb.log({"epoch": epoch, "train_loss": train_loss,
                            "train_acc": train_acc, 
                            "train_epoch_time": t1,
                            "time_ratio_optim": time_ratio_optim,
                            "true_optim_time": t_optim_cumul})

            if self.use_scheduler_:
                scheduler.step()

            test_dummy = False
            if test_dummy:
                t0 = time.time()

                acc = QuickAccuracyMetric()

                estimator_.eval()
                with torch.no_grad():
                    for _, elem in enumerate(test_loader):
                        data, target = io.split_data_target(elem, self.device)

                        output = estimator_(*data)

                        acc.update(output, target)

                val_acc = acc.compute(mode="all")

                t1 = time.time() - t0

                print("Epoch {} (dummy) | Val acc {:.2f} | Time {:.1f} seconds.".format(
                    epoch+1, val_acc, t1))

                if self.use_wandb:
                    wandb.log({"epoch": epoch+1,
                               "val_acc": val_acc,
                               "val_epoch_time": t1})

        # ====================================================================
        #                        Generate the ensemble
        # ====================================================================

        # Set the internal optimizer
        estimator_.zero_grad()
        optimizer = set_module.set_optimizer(
            estimator_, self.optimizer_name, **self.optimizer_args
        )

        # Utils
        best_acc = - 1
        best_acc_epoch = - 1
        n_iters = len(train_loader)
        updated = False
        epoch = 0

        iteration_idx = 1

        while len(self.estimators_) < self.n_estimators:

            # Training
            estimator_.train()

            t0 = time.time()
            t_optim_cumul = 0
            t_update_ensemble_and_test = 0
            acc = QuickAccuracyMetric()
            train_loss = 0

            for elem in train_loader:

                # Update learning rate
                self._adjust_lr(
                    optimizer, iteration_idx, iterations_per_estimator, lr_1, lr_2
                )

                iteration_idx = iteration_idx + 1

                data, target = io.split_data_target(elem, self.device)
                #batch_size = data[0].size(0)

                t0_optim = time.time()

                optimizer.zero_grad()
                output = estimator_(*data)
                loss = self._criterion(output, target)
                loss.backward()
                optimizer.step()

                t_optim_cumul += time.time() - t0_optim
                train_loss += loss.item()
                acc.update(output, target)

                total_iters += 1

                """ 
                epoch_fine_grained = epoch+dummy_epochs + \
                    float(batch_idx+1)/len(train_loader)

                if self.use_wandb:
                    wandb.log(
                        {"epoch": epoch_fine_grained,
                         "real_time_learning_rate": optimizer.param_groups[0]["lr"]})
                """

                t0_update_ensemble_and_test = time.time()

                iteration_idx, iterations_per_estimator, estimator_, \
                    test_loader, epoch, save_model, save_dir, dummy_epochs, \
                    best_acc, best_acc_epoch, n_iters, updated, total_iters = \
                    self.update_ensemble_and_test(iteration_idx,
                                                  iterations_per_estimator,
                                                  estimator_,
                                                  test_loader,
                                                  epoch,
                                                  save_model,
                                                  save_dir,
                                                  dummy_epochs,
                                                  best_acc,
                                                  best_acc_epoch,
                                                  n_iters,
                                                  updated,
                                                  total_iters)
                t_update_ensemble_and_test += time.time() - t0_update_ensemble_and_test

            t1 = time.time() - t0 - t_update_ensemble_and_test
            time_ratio_optim = t_optim_cumul / t1
            train_loss /= len(train_loader)
            train_acc = acc.compute()
            
            print("Epoch {} | Train loss {:.2f} | Train acc {:.2f} | Time {:.1f} ({:.1f}) seconds. Optimization time ratio: {:.2f}. UpdateEnsembleAndTest: {:.1f} seconds.".format(
                epoch+1 +
                dummy_epochs, train_loss,
                train_acc, t1, t_optim_cumul, time_ratio_optim, t_update_ensemble_and_test))
            if self.use_wandb:
                wandb.log({"epoch": epoch+1+dummy_epochs, "train_loss": train_loss,
                           "train_acc": train_acc,
                           "train_epoch_time": t1,
                           "time_ratio_optim": time_ratio_optim,
                           "true_optim_time": t_optim_cumul, 
                           "t_update_ensemble_and_test": t_update_ensemble_and_test})

            epoch += 1

        if save_model and not test_loader:
            io.save(self, save_dir)














































@torchensemble_model_doc(
    """Implementation on the FastGeometricRegressor.""", "seq_model"
)
class FastGeometricRegressor(_BaseFastGeometric, BaseRegressor):
    @torchensemble_model_doc(
        """Implementation on the data forwarding in FastGeometricRegressor.""",  # noqa: E501
        "regressor_forward",
    )
    def forward(self, *x):
        pred = self._forward(*x)
        return pred

    @torchensemble_model_doc(
        (
            """Set the attributes on optimizer for FastGeometricRegressor. """
            + """Notice that keyword arguments specified here will also be """
            + """used in the ensembling stage except the learning rate."""
        ),
        "set_optimizer",
    )
    def set_optimizer(self, optimizer_name, **kwargs):
        super().set_optimizer(optimizer_name=optimizer_name, **kwargs)

    @torchensemble_model_doc(
        (
            """Set the attributes on scheduler for FastGeometricRegressor. """
            + """Notice that this scheduler will only be used in the stage on """  # noqa: E501
            + """fitting the dummy base estimator."""
        ),
        "set_scheduler",
    )
    def set_scheduler(self, scheduler_name, **kwargs):
        super().set_scheduler(scheduler_name=scheduler_name, **kwargs)

    @torchensemble_model_doc(
        """Set the training criterion for FastGeometricRegressor.""",
        "set_criterion",
    )
    def set_criterion(self, criterion):
        super().set_criterion(criterion)

    @_fast_geometric_model_doc(
        """Implementation on the training stage of FastGeometricRegressor.""",  # noqa: E501
        "fit",
    )
    def fit(
        self,
        train_loader,
        cycle=4,
        lr_1=5e-2,
        lr_2=1e-4,
        epochs=100,
        log_interval=100,
        test_loader=None,
        save_model=True,
        save_dir=None,
    ):
        self._validate_parameters(epochs, log_interval)
        self.n_outputs = self._decide_n_outputs(train_loader)

        # ====================================================================
        #                Train the dummy estimator (estimator_)
        # ====================================================================

        estimator_ = self._make_estimator()

        # Set the optimizer and scheduler
        optimizer = set_module.set_optimizer(
            estimator_, self.optimizer_name, **self.optimizer_args
        )

        if self.use_scheduler_:
            scheduler = set_module.set_scheduler(
                optimizer, self.scheduler_name, **self.scheduler_args
            )

        # Check the training criterion
        if not hasattr(self, "_criterion"):
            self._criterion = nn.MSELoss()

        # Utils
        total_iters = 0

        for epoch in range(epochs):

            # Training
            estimator_.train()
            
            for batch_idx, elem in enumerate(train_loader):

                data, target = io.split_data_target(elem, self.device)

                optimizer.zero_grad()
                output = estimator_(*data)
                loss = self._criterion(output, target)
                loss.backward()
                optimizer.step()

                # Print training status
                if batch_idx % log_interval == 0:
                    with torch.no_grad():
                        msg = "Epoch: {:03d} | Batch: {:03d} | Loss: {:.5f}"
                        print(msg.format(epoch, batch_idx, loss))
                        if self.tb_logger:
                            self.tb_logger.add_scalar(
                                "fast_geometric/Base_Est/Train_Loss",
                                loss,
                                total_iters,
                            )
                total_iters += 1

            if self.use_scheduler_:
                scheduler.step()

        # ====================================================================
        #                        Generate the ensemble
        # ====================================================================

        # Set the internal optimizer
        estimator_.zero_grad()
        optimizer = set_module.set_optimizer(
            estimator_, self.optimizer_name, **self.optimizer_args
        )

        # Utils
        best_loss = float("inf")
        n_iters = len(train_loader)
        updated = False
        epoch = 0
        
        

        while len(self.estimators_) < self.n_estimators:

            # Training
            estimator_.train()
            for batch_idx, elem in enumerate(train_loader):

                # Update learning rate
                self._adjust_lr(
                    optimizer, epoch, batch_idx, n_iters, cycle, lr_1, lr_2
                )

                data, target = io.split_data_target(elem, self.device)

                optimizer.zero_grad()
                output = estimator_(*data)
                loss = self._criterion(output, target)
                loss.backward()
                optimizer.step()

                # Print training status
                if batch_idx % log_interval == 0:
                    with torch.no_grad():
                        msg = "Epoch: {:03d} | Batch: {:03d} | Loss: {:.5f}"
                        print(msg.format(epoch, batch_idx, loss))
                        if self.tb_logger:
                            self.tb_logger.add_scalar(
                                "fast_geometric/Ensemble-Est_{}".format(
                                    len(self.estimators_)
                                )
                                + "/Train_Loss",
                                loss,
                                total_iters,
                            )
                total_iters += 1

            # Update the ensemble
            if (epoch % cycle + 1) == cycle // 2:
                base_estimator = self._make_estimator()
                base_estimator.load_state_dict(estimator_.state_dict())
                self.estimators_.append(base_estimator)
                updated = True
                total_iters = 0

                msg = "Save the base estimator with index: {}"
                print(msg.format(len(self.estimators_) - 1))

            # Validation after each base estimator being added
            if test_loader and updated:
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
                        "Epoch: {:03d} | Validation Loss: {:.5f} |"
                        " Historical Best: {:.5f}"
                    )
                    print(msg.format(epoch, val_loss, best_loss))
                    
                updated = False  # reset the updating flag
            epoch += 1

        if save_model and not test_loader:
            io.save(self, save_dir)

    @torchensemble_model_doc(item="regressor_evaluate")
    def evaluate(self, test_loader):
        return super().evaluate(test_loader)

    @torchensemble_model_doc(item="predict")
    def predict(self, *x):
        return super().predict(*x)
