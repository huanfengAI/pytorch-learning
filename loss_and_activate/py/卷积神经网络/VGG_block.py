import torch
from torch import nn

def vgg_block(num_convs,in_channels,out_channels):
	net=[nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),nn.ReLU(inplace=True)]
	for i in range(num_convs-1):
		net.append(nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1))
		net.append(nn.ReLU(inplace=True))
	net.append(nn.MaxPool2d(2,2))
	return nn.Sequential(*net)#*net是不要列表只要列表中的元素
	
#这里我明白一个道理，这个道理就是，卷积层什么的没有必要非得定义在类里面，定义在类里面只是为了调用方便，我们可以直接定义一个卷积层，就可以直接使用，或者定义很多卷基层还有激活层，这样可以使用sequential来将它们组合起来

block_demo=vgg_block(3,64,128)#定义一个三个层的，输入通道是64，输出通道是128
print(block_demo)
input_demo=torch.zeros(1,64,300,300)#定义一张图片，图片的通道数为64，长和宽为300
output_demo=block_demo(input_demo)
print(output_demo.shape)#torch.Size([1, 128, 150, 150])

