import torch
from torch import nn

def vgg_block(num_convs,in_channels,out_channels):
	net=[nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),nn.ReLU(inplace=True)]
	for i in range(num_convs-1):
		net.append(nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1))
		net.append(nn.ReLU(inplace=True))
	net.append(nn.MaxPool2d(2,2))
	return nn.Sequential(*net)#*net是不要列表只要列表中的元素