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
    

class inception(nn.Module):
	def __init__(self,in_channel,out1_1,out2_1,out2_3,out3_1,out3_5,out4_1):
		super(inception,self).__init__()
		self.branch1x1=conv_relu(in_channel,out1_1,1)#1*1的卷积核
		
		self.branch3x3=nn.Sequential(
			conv_relu(in_channel,out2_1,1),#1*1的卷积核，然后是3*3的卷积核
			conv_relu(out2_1,out2_3,3,padding=1)
		)
		self.branch5x5=nn.Sequential(#1*1的卷积核，然后是5*5的卷积核
			conv_relu(in_channel,out3_1,1),
			conv_relu(out3_1,out3_5,5,padding=2)
		)
		self.branch_pool=nn.Sequential(#3*3的池化层，然后是1*1的卷积核
			nn.MaxPool2d(3,stride=1,padding=1),
			conv_relu(in_channel,out4_1,1)
		)
		
	def forward(self,x):
			f1 = self.branch1x1(x)
			f2 = self.branch3x3(x)
			f3 = self.branch5x5(x)
			f4 = self.branch_pool(x)
			output = torch.cat((f1, f2, f3, f4), dim=1)#在通道上连接起来
			return output
   
test_net =inception(3, 64, 48, 64, 64, 96, 32)
test_x = torch.zeros(1, 3, 96, 96)
test_y = test_net(test_x)
print(test_y.shape)#torch.Size([1, 256, 96, 96])
