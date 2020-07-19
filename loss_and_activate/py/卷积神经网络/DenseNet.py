import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
#denseNet的基本结构是图片大小不变的卷积操作，也就是卷积核为3，填充为1的卷积操作
#然后每次经过卷积操作之后，输入通道和输出通道会经过growth_rate的速率发生变化
def conv_block(in_channel, out_channel):
    layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(True),
        nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False)
    )
    return layer

class dense_block(nn.Module):
    def __init__(self, in_channel, growth_rate, num_layers):
        super(dense_block, self).__init__()
        block = []
        channel = in_channel
        for i in range(num_layers):
            print("channel",channel)
            print("growth_rate",growth_rate)
            block.append(conv_block(channel, growth_rate))
            channel += growth_rate
            
        self.net = nn.Sequential(*block)
        
    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            print("x",x.shape)
            print("out",out.shape)
            x = torch.cat((out, x), dim=1)
        return x

test_net = dense_block(3, 12, 3)
#print(test_net)
test_x = torch.zeros(1, 3, 96, 96)
print('input shape: {} x {} x {}'.format(test_x.shape[1], test_x.shape[2], test_x.shape[3]))
test_y = test_net(test_x)
print('output shape: {} x {} x {}'.format(test_y.shape[1], test_y.shape[2], test_y.shape[3]))
