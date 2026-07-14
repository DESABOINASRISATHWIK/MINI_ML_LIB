import numpy as np
from .autograd import Value
from .modules.base import Module

class BCELoss(Module):
    """
    Binary Cross-Entropy Loss:
        L = -[y*log(p) + (1 - y)*log(1 - p)]
    where p is predicted probability (after sigmoid).
    """

    def call(self, y_pred, y_true):
        eps = 1e-8
        y_pred_clamped = Value(np.clip(y_pred.data, eps, 1 - eps))
        loss = -(y_true * y_pred_clamped.log() + (1 - y_true) * (1 - y_pred_clamped).log()).mean()
        return loss

 

class CrossEntropyLoss(Module):
    def call(self, logits, y_true):
        N = len(y_true)
        losses = []
        for i in range(N):
            exps = [l.exp() for l in logits[i]]
            denom = sum(exps)
            probs = [e / denom for e in exps]
            correct_prob = probs[y_true[i]]
            losses.append((-correct_prob.log()))
        return sum(losses) * (1.0 / N)




