import torch
import torch.nn as nn

import os
import logging
# import torch.legacy.nn as lnn

from functools import reduce
from torch.autograd import Variable



BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func, self.forward_prepare(input)))


class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func, self.forward_prepare(input))


class PoseResneXt(nn.Module):
    def __init__(self):
        self.inplanes = 2048
        self.deconv_with_bias = False

        super(PoseResneXt, self).__init__()
        self.resnext_101_64x4d = nn.Sequential(  # Sequential,
            nn.Conv2d(3, 64, (7, 7), (2, 2), (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (2, 2), (1, 1)),
            nn.Sequential(  # Sequential,
                nn.Sequential(  # Sequential,
                    LambdaMap(lambda x: x,  # ConcatTable,
                              nn.Sequential(  # Sequential,
                                  nn.Sequential(  # Sequential,
                                      nn.Conv2d(64, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 64, bias=False),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(),
                                  ),
                                  nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(256),
                              ),
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(64, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(256),
                              ),
                              ),
                    LambdaReduce(lambda x, y: x + y),  # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential(  # Sequential,
                    LambdaMap(lambda x: x,  # ConcatTable,
                              nn.Sequential(  # Sequential,
                                  nn.Sequential(  # Sequential,
                                      nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 64, bias=False),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(),
                                  ),
                                  nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(256),
                              ),
                              Lambda(lambda x: x),  # Identity,
                              ),
                    LambdaReduce(lambda x, y: x + y),  # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential(  # Sequential,
                    LambdaMap(lambda x: x,  # ConcatTable,
                              nn.Sequential(  # Sequential,
                                  nn.Sequential(  # Sequential,
                                      nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 64, bias=False),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(),
                                  ),
                                  nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(256),
                              ),
                              Lambda(lambda x: x),  # Identity,
                              ),
                    LambdaReduce(lambda x, y: x + y),  # CAddTable,
                    nn.ReLU(),
                ),
            ),
            nn.Sequential(  # Sequential,
                nn.Sequential(  # Sequential,
                    LambdaMap(lambda x: x,  # ConcatTable,
                              nn.Sequential(  # Sequential,
                                  nn.Sequential(  # Sequential,
                                      nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(),
                                      nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1), 1, 64, bias=False),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(),
                                  ),
                                  nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(512),
                              ),
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(256, 512, (1, 1), (2, 2), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(512),
                              ),
                              ),
                    LambdaReduce(lambda x, y: x + y),  # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential(  # Sequential,
                    LambdaMap(lambda x: x,  # ConcatTable,
                              nn.Sequential(  # Sequential,
                                  nn.Sequential(  # Sequential,
                                      nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(),
                                      nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 64, bias=False),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(),
                                  ),
                                  nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(512),
                              ),
                              Lambda(lambda x: x),  # Identity,
                              ),
                    LambdaReduce(lambda x, y: x + y),  # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential(  # Sequential,
                    LambdaMap(lambda x: x,  # ConcatTable,
                              nn.Sequential(  # Sequential,
                                  nn.Sequential(  # Sequential,
                                      nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(),
                                      nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 64, bias=False),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(),
                                  ),
                                  nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(512),
                              ),
                              Lambda(lambda x: x),  # Identity,
                              ),
                    LambdaReduce(lambda x, y: x + y),  # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential(  # Sequential,
                    LambdaMap(lambda x: x,  # ConcatTable,
                              nn.Sequential(  # Sequential,
                                  nn.Sequential(  # Sequential,
                                      nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(),
                                      nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 64, bias=False),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(),
                                  ),
                                  nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(512),
                              ),
                              Lambda(lambda x: x),  # Identity,
                              ),
                    LambdaReduce(lambda x, y: x + y),  # CAddTable,
                    nn.ReLU(),
                ),
            ),
            nn.Sequential(  # Sequential,
                nn.Sequential(  # Sequential,
                    LambdaMap(lambda x: x,  # ConcatTable,
                              nn.Sequential(  # Sequential,
                                  nn.Sequential(  # Sequential,
                                      nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                      nn.Conv2d(1024, 1024, (3, 3), (2, 2), (1, 1), 1, 64, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                  ),
                                  nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(1024),
                              ),
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(512, 1024, (1, 1), (2, 2), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(1024),
                              ),
                              ),
                    LambdaReduce(lambda x, y: x + y),  # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential(  # Sequential,
                    LambdaMap(lambda x: x,  # ConcatTable,
                              nn.Sequential(  # Sequential,
                                  nn.Sequential(  # Sequential,
                                      nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                      nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                  ),
                                  nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(1024),
                              ),
                              Lambda(lambda x: x),  # Identity,
                              ),
                    LambdaReduce(lambda x, y: x + y),  # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential(  # Sequential,
                    LambdaMap(lambda x: x,  # ConcatTable,
                              nn.Sequential(  # Sequential,
                                  nn.Sequential(  # Sequential,
                                      nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                      nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                  ),
                                  nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(1024),
                              ),
                              Lambda(lambda x: x),  # Identity,
                              ),
                    LambdaReduce(lambda x, y: x + y),  # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential(  # Sequential,
                    LambdaMap(lambda x: x,  # ConcatTable,
                              nn.Sequential(  # Sequential,
                                  nn.Sequential(  # Sequential,
                                      nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                      nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                  ),
                                  nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(1024),
                              ),
                              Lambda(lambda x: x),  # Identity,
                              ),
                    LambdaReduce(lambda x, y: x + y),  # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential(  # Sequential,
                    LambdaMap(lambda x: x,  # ConcatTable,
                              nn.Sequential(  # Sequential,
                                  nn.Sequential(  # Sequential,
                                      nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                      nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                  ),
                                  nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(1024),
                              ),
                              Lambda(lambda x: x),  # Identity,
                              ),
                    LambdaReduce(lambda x, y: x + y),  # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential(  # Sequential,
                    LambdaMap(lambda x: x,  # ConcatTable,
                              nn.Sequential(  # Sequential,
                                  nn.Sequential(  # Sequential,
                                      nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                      nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                  ),
                                  nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(1024),
                              ),
                              Lambda(lambda x: x),  # Identity,
                              ),
                    LambdaReduce(lambda x, y: x + y),  # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential(  # Sequential,
                    LambdaMap(lambda x: x,  # ConcatTable,
                              nn.Sequential(  # Sequential,
                                  nn.Sequential(  # Sequential,
                                      nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                      nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                  ),
                                  nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(1024),
                              ),
                              Lambda(lambda x: x),  # Identity,
                              ),
                    LambdaReduce(lambda x, y: x + y),  # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential(  # Sequential,
                    LambdaMap(lambda x: x,  # ConcatTable,
                              nn.Sequential(  # Sequential,
                                  nn.Sequential(  # Sequential,
                                      nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                      nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                  ),
                                  nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(1024),
                              ),
                              Lambda(lambda x: x),  # Identity,
                              ),
                    LambdaReduce(lambda x, y: x + y),  # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential(  # Sequential,
                    LambdaMap(lambda x: x,  # ConcatTable,
                              nn.Sequential(  # Sequential,
                                  nn.Sequential(  # Sequential,
                                      nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                      nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                  ),
                                  nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(1024),
                              ),
                              Lambda(lambda x: x),  # Identity,
                              ),
                    LambdaReduce(lambda x, y: x + y),  # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential(  # Sequential,
                    LambdaMap(lambda x: x,  # ConcatTable,
                              nn.Sequential(  # Sequential,
                                  nn.Sequential(  # Sequential,
                                      nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                      nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                  ),
                                  nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(1024),
                              ),
                              Lambda(lambda x: x),  # Identity,
                              ),
                    LambdaReduce(lambda x, y: x + y),  # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential(  # Sequential,
                    LambdaMap(lambda x: x,  # ConcatTable,
                              nn.Sequential(  # Sequential,
                                  nn.Sequential(  # Sequential,
                                      nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                      nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                  ),
                                  nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(1024),
                              ),
                              Lambda(lambda x: x),  # Identity,
                              ),
                    LambdaReduce(lambda x, y: x + y),  # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential(  # Sequential,
                    LambdaMap(lambda x: x,  # ConcatTable,
                              nn.Sequential(  # Sequential,
                                  nn.Sequential(  # Sequential,
                                      nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                      nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                  ),
                                  nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(1024),
                              ),
                              Lambda(lambda x: x),  # Identity,
                              ),
                    LambdaReduce(lambda x, y: x + y),  # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential(  # Sequential,
                    LambdaMap(lambda x: x,  # ConcatTable,
                              nn.Sequential(  # Sequential,
                                  nn.Sequential(  # Sequential,
                                      nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                      nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                  ),
                                  nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(1024),
                              ),
                              Lambda(lambda x: x),  # Identity,
                              ),
                    LambdaReduce(lambda x, y: x + y),  # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential(  # Sequential,
                    LambdaMap(lambda x: x,  # ConcatTable,
                              nn.Sequential(  # Sequential,
                                  nn.Sequential(  # Sequential,
                                      nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                      nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                  ),
                                  nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(1024),
                              ),
                              Lambda(lambda x: x),  # Identity,
                              ),
                    LambdaReduce(lambda x, y: x + y),  # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential(  # Sequential,
                    LambdaMap(lambda x: x,  # ConcatTable,
                              nn.Sequential(  # Sequential,
                                  nn.Sequential(  # Sequential,
                                      nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                      nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                  ),
                                  nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(1024),
                              ),
                              Lambda(lambda x: x),  # Identity,
                              ),
                    LambdaReduce(lambda x, y: x + y),  # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential(  # Sequential,
                    LambdaMap(lambda x: x,  # ConcatTable,
                              nn.Sequential(  # Sequential,
                                  nn.Sequential(  # Sequential,
                                      nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                      nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                  ),
                                  nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(1024),
                              ),
                              Lambda(lambda x: x),  # Identity,
                              ),
                    LambdaReduce(lambda x, y: x + y),  # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential(  # Sequential,
                    LambdaMap(lambda x: x,  # ConcatTable,
                              nn.Sequential(  # Sequential,
                                  nn.Sequential(  # Sequential,
                                      nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                      nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                  ),
                                  nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(1024),
                              ),
                              Lambda(lambda x: x),  # Identity,
                              ),
                    LambdaReduce(lambda x, y: x + y),  # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential(  # Sequential,
                    LambdaMap(lambda x: x,  # ConcatTable,
                              nn.Sequential(  # Sequential,
                                  nn.Sequential(  # Sequential,
                                      nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                      nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                  ),
                                  nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(1024),
                              ),
                              Lambda(lambda x: x),  # Identity,
                              ),
                    LambdaReduce(lambda x, y: x + y),  # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential(  # Sequential,
                    LambdaMap(lambda x: x,  # ConcatTable,
                              nn.Sequential(  # Sequential,
                                  nn.Sequential(  # Sequential,
                                      nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                      nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                  ),
                                  nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(1024),
                              ),
                              Lambda(lambda x: x),  # Identity,
                              ),
                    LambdaReduce(lambda x, y: x + y),  # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential(  # Sequential,
                    LambdaMap(lambda x: x,  # ConcatTable,
                              nn.Sequential(  # Sequential,
                                  nn.Sequential(  # Sequential,
                                      nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                      nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                  ),
                                  nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(1024),
                              ),
                              Lambda(lambda x: x),  # Identity,
                              ),
                    LambdaReduce(lambda x, y: x + y),  # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential(  # Sequential,
                    LambdaMap(lambda x: x,  # ConcatTable,
                              nn.Sequential(  # Sequential,
                                  nn.Sequential(  # Sequential,
                                      nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                      nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                  ),
                                  nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(1024),
                              ),
                              Lambda(lambda x: x),  # Identity,
                              ),
                    LambdaReduce(lambda x, y: x + y),  # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential(  # Sequential,
                    LambdaMap(lambda x: x,  # ConcatTable,
                              nn.Sequential(  # Sequential,
                                  nn.Sequential(  # Sequential,
                                      nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                      nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                  ),
                                  nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(1024),
                              ),
                              Lambda(lambda x: x),  # Identity,
                              ),
                    LambdaReduce(lambda x, y: x + y),  # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential(  # Sequential,
                    LambdaMap(lambda x: x,  # ConcatTable,
                              nn.Sequential(  # Sequential,
                                  nn.Sequential(  # Sequential,
                                      nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                      nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias=False),
                                      nn.BatchNorm2d(1024),
                                      nn.ReLU(),
                                  ),
                                  nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(1024),
                              ),
                              Lambda(lambda x: x),  # Identity,
                              ),
                    LambdaReduce(lambda x, y: x + y),  # CAddTable,
                    nn.ReLU(),
                ),
            ),
            nn.Sequential(  # Sequential,
                nn.Sequential(  # Sequential,
                    LambdaMap(lambda x: x,  # ConcatTable,
                              nn.Sequential(  # Sequential,
                                  nn.Sequential(  # Sequential,
                                      nn.Conv2d(1024, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                      nn.BatchNorm2d(2048),
                                      nn.ReLU(),
                                      nn.Conv2d(2048, 2048, (3, 3), (2, 2), (1, 1), 1, 64, bias=False),
                                      nn.BatchNorm2d(2048),
                                      nn.ReLU(),
                                  ),
                                  nn.Conv2d(2048, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(2048),
                              ),
                              nn.Sequential(  # Sequential,
                                  nn.Conv2d(1024, 2048, (1, 1), (2, 2), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(2048),
                              ),
                              ),
                    LambdaReduce(lambda x, y: x + y),  # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential(  # Sequential,
                    LambdaMap(lambda x: x,  # ConcatTable,
                              nn.Sequential(  # Sequential,
                                  nn.Sequential(  # Sequential,
                                      nn.Conv2d(2048, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                      nn.BatchNorm2d(2048),
                                      nn.ReLU(),
                                      nn.Conv2d(2048, 2048, (3, 3), (1, 1), (1, 1), 1, 64, bias=False),
                                      nn.BatchNorm2d(2048),
                                      nn.ReLU(),
                                  ),
                                  nn.Conv2d(2048, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(2048),
                              ),
                              Lambda(lambda x: x),  # Identity,
                              ),
                    LambdaReduce(lambda x, y: x + y),  # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential(  # Sequential,
                    LambdaMap(lambda x: x,  # ConcatTable,
                              nn.Sequential(  # Sequential,
                                  nn.Sequential(  # Sequential,
                                      nn.Conv2d(2048, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                      nn.BatchNorm2d(2048),
                                      nn.ReLU(),
                                      nn.Conv2d(2048, 2048, (3, 3), (1, 1), (1, 1), 1, 64, bias=False),
                                      nn.BatchNorm2d(2048),
                                      nn.ReLU(),
                                  ),
                                  nn.Conv2d(2048, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias=False),
                                  nn.BatchNorm2d(2048),
                              ),
                              Lambda(lambda x: x),  # Identity,
                              ),
                    LambdaReduce(lambda x, y: x + y),  # CAddTable,
                    nn.ReLU(),
                ),
            ),
            # nn.AvgPool2d((7, 7), (1, 1)),
            # Lambda(lambda x: x.view(x.size(0), -1)),  # View,
            # nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(2048, 1000)),
            # Linear,
        )

        self.deconv_layers = self._make_deconv_layer(
            3, [256, 256, 256], [4, 4, 4], )

        self.final_layer = nn.Conv2d(
            in_channels=256,
            out_channels=15,
            kernel_size=1,
            stride=1,
            padding=0)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=0.1))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.resnext_101_64x4d(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)

        return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            pretrained_state_dict = torch.load(pretrained)
            need_init_state_dict = {}
            logger.info('=> loading pretrained model {}'.format(pretrained))
            for name, value in pretrained_state_dict.items():
                need_init_state_dict['resnext_101_64x4d.{}'.format(name)] = value
            self.load_state_dict(need_init_state_dict, strict=False)
        else:
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                    # nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)


def get_pose_net(is_train, **kwargs):
    pretrained = './models/pytorch/resnext_101_64x4d.pth'
    model = PoseResneXt()

    if is_train:
        model.init_weights(pretrained)

    return model


def main():
    model = PoseResneXt()

    device = torch.device('cuda:0')
    pretrained_dict = torch.load('../../models/pytorch/resnext_101_64x4d.pth')
    need_init_state_dict = {}

    for name, value in pretrained_dict.items():
        need_init_state_dict['resnext_101_64x4d.{}'.format(name)] = value
    for name, value in need_init_state_dict.items():
        print(name, ':', value)
    print(need_init_state_dict)

    model.load_state_dict(need_init_state_dict, strict=False)
    model.to(device)
    x = torch.rand(1, 3, 256, 192).to(device)
    x = model(x)
    print(x.size())

    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())


if __name__ == '__main__':
    main()
