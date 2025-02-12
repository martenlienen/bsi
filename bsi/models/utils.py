import torch.nn as nn


def actfn_from_str(name: str):
    actfns = {
        "silu": nn.SiLU,
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "softplus": nn.Softplus,
        "tanh": nn.Tanh,
    }
    return actfns[name]
