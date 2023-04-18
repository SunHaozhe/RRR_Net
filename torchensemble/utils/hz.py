import math
import numpy as np


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
            self.correct_cnt[k] += (current_class *
                                    correct_prediction).sum().item()
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
        elif mode == "class_weighted":
            raise NotImplementedError
            accuracy_per_class = []
            support_per_class = []
            total_nb_examples = np.sum(self.total_cnt)
            for k in range(self.nb_classes):
                if self.total_cnt[k] == 0:
                    continue
                else:
                    accuracy_per_class.append(
                        self.correct_cnt[k] / self.total_cnt[k])
                support_per_class.append(self.total_cnt[k] / total_nb_examples)
            accuracy_per_class = np.asarray(accuracy_per_class)
            support_per_class = np.asarray(support_per_class)
            res = np.mean(np.multiply(accuracy_per_class, support_per_class))
        elif mode == "worst_classes":
            """
            takes into account classes that have 0 support
            """
            nb_worst_classes = max(1, math.floor(
                ratio_worst_classes * self.nb_classes))
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
