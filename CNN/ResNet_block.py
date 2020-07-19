import numpy as np
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torch import nn
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from datetime import datetime
from torchvision.datasets import CIFAR10
import torch.nn.functional as F


def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)

class Resnet_block(nn.Module):
	def __init__(self,in_channel,out_channel,same_shape=True):
		super(Resnet_block,self).__init__()
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