import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
#如果same_shape=True那么步长就是1，那么此时输入图片多么大输出图片就多么大，因为填充是1，卷积核的大小是3。如果same_shape=False,那么步长就是2，那么此时的在卷积核为3，填充为1的情况下，图片会缩小为原来的两倍，那么此时x和out就不一般大了，就不可以相加了，所以我们要对x做缩小操作，具体来说我们需要使用卷积核大小为1步长为2的做卷积操作，这样也会将x缩小两倍就可以使得x和out的长和宽一致了
def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)

class Resnet(nn.Module):
	def __init__(self,in_channel,out_channel,same_shape=True):
		super(Resnet,self).__init__()
		self.same_shape=same_shape
		stride=1 if self.same_shape else 2
		
		self.conv1=conv3x3(in_channel,out_channel,stride=stride)
		self.bn1=nn.BatchNorm2d(out_channel)

		self.conv2=conv3x3(out_channel,out_channel)
		self.bn2=nn.BatchNorm2d(out_channel)
		if not self.same_shape:
			self.conv3 =nn.Conv2d(in_channel,out_channel,1,stride=stride)
	def forward(self,x):
		out=self.conv1(x)
		out=F.relu(self.bn1(out),True)
		out=self.conv2(out)
		out=F.relu(self.bn2(out),True)

		if not self.same_shape:
			x=self.conv3(x)
		
		return F.relu(x+out,True)

test_net = Resnet(32, 32)
test_x = Variable(torch.zeros(1, 32, 96, 96))
print('input: {}'.format(test_x.shape))
test_y = test_net(test_x)
print('output: {}'.format(test_y.shape))
print("----------------------------------------------")
test_net = Resnet(3, 32, False)
test_x = Variable(torch.zeros(1, 3, 96, 96))
print('input: {}'.format(test_x.shape))
test_y = test_net(test_x)
print('output: {}'.format(test_y.shape))

