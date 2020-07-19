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
class Resnet(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(Resnet, self).__init__()
        self.block1 = nn.Conv2d(in_channel, 64, 7, 2)
        self.block2 = nn.Sequential(
            nn.MaxPool2d(3, 2),
            Resnet_block(64, 64),
            Resnet_block(64, 64)
        )
        
        self.block3 = nn.Sequential(
            Resnet_block(64, 128, False),
            Resnet_block(128, 128)
        )
        
        self.block4 = nn.Sequential(
            Resnet_block(128, 256, False),
            Resnet_block(256, 256)
        )
        
        self.block5 = nn.Sequential(
            Resnet_block(256, 512, False),
            Resnet_block(512, 512),
            nn.AvgPool2d(3)
        )
        
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
        
        
net=Resnet(3,10)