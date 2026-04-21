import torch
import torch.nn as nn

from src.models.TFT.domain_classifier import TFTDomainClassifier

class TFTGRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def dann_lambda_schedule(p: float) ->float:
    import math
    return float(2.0/ (1.0 + math.exp(-10.0 * float(p))) - 1.0)

