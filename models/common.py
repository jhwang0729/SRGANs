from typing import Union

from torch import nn


class Conv(nn.Module):
    def __init__(self,
                 c1: int,
                 c2: int,
                 k: int,
                 s: int,
                 act: Union[nn.Module, None] = None):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is None else act

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWSampleConv(nn.Module):
    def __init__(self,
                 c1: int,
                 c2: int):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 3, 2, 1)
        self.identity = Conv(c1, c2, 1, 2)





class RepConv(nn.Module):
    def __init__(self,
                 c1: int,
                 c2: int,
                 k: int = 3,
                 s: int = 1,
                 act: Union[nn.Module, None] = None):
        super().__init__()
        self.conv1 = Conv()


class RepResBlcok(nn.Module):
    def __init__(self,
                 c1: int,
                 c2: int,
                 k: int,
                 s: int,
                 act: Union[nn.Module, None] = None):
        super().__init__()
        self.conv = Conv(c1, c2, k, s, act)