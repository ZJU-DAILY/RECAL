import torch.nn as nn
from abc import abstractmethod


class BaseModel(nn.Module):
    def __init__(
        self, input_dim, output_dim, seq_len=12, horizon=12, node_num=None, **kwargs
    ):
        super(BaseModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.horizon = horizon
        if node_num is not None:
            self.node_num = node_num

    @abstractmethod
    def forward(self):  # (x, label=None, adj=None)
        raise NotImplementedError

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])
