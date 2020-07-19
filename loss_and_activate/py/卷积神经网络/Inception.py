import sys
sys.path.append('..')

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
def conv_relu(in_channel, out_channel, kernel, stride=1, padding=0):
    layer = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel, stride, padding),
        nn.BatchNorm2d(out_channel, eps=1e-3),
        nn.ReLU(True)
    )
    return layer
    
#in_channel=3
#out1_1=64
#out2_1=48
#out2_3=64
#out3_1=64
#out3_5=96
#out4_1=32  
class inception(nn.Module):
	def __init__(self,in_channel,out1_1,out2_1,out2_3,out3_1,out3_5,out4_1):
		super(inception,self).__init__()
		self.branch1x1=conv_relu(in_channel,out1_1,1)
		
		self.branch3x3=nn.Sequential(
			conv_relu(in_channel,out2_1,1),
			conv_relu(out2_1,out2_3,3,padding=1)
		)
		self.branch5x5=nn.Sequential(
			conv_relu(in_channel,out3_1,1),
			conv_relu(out3_1,out3_5,5,padding=2)
		)
		self.branch_pool=nn.Sequential(
			nn.MaxPool2d(3,stride=1,padding=1),
			conv_relu(in_channel,out4_1,1)
		)
		
	def forward(self,x):
			f1 = self.branch1x1(x)
			f2 = self.branch3x3(x)
			f3 = self.branch5x5(x)
			f4 = self.branch_pool(x)
			output = torch.cat((f1, f2, f3, f4), dim=1)
			return output
   
test_net =inception(3, 64, 48, 64, 64, 96, 32)
test_x = torch.zeros(1, 3, 96, 96)
print('input shape: {} x {} x {}'.format(test_x.shape[1], test_x.shape[2], test_x.shape[3]))
test_y = test_net(test_x)
print('output shape: {} x {} x {}'.format(test_y.shape[1], test_y.shape[2], test_y.shape[3]))
