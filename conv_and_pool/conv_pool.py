from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
im=Image.open('./cat').convert('L')#转成灰度图，通道为1
im=np.array(im,dtype='float32')#转成numpy[121,121]
im=torch.from_numpy(im.reshape((1,1,im.shape[0],im.shape[1])))#转成tensor[1,1,121,121]

#############卷积操作################
conv1=nn.Conv2d(1,1,3,bias=False)#卷积层
sobel_kernel=np.array([[-1,-1,-1],[-1,-8,-1],[-1,-1,-1]],dtype='float32')#卷积和参数
sobel_kernel=sobel_kernel.reshape((1,1,3,3))#转成卷积核需要的参数维度
conv1.weight.data=torch.from_numpy(sobel_kernel)#为卷积层参数赋值
edge=conv1(im)#卷积操作[1,1,60,60]
edge=edge.data.squeeze().numpy()#转成numpy,[60,60]
plt.imshow(edge,cmap="gray")#显示卷积之后的图片
#########池化操作############
pool1=nn.MaxPool2d(2,2)#定义池化层
small_img=pool1(im)#池化#torch.Size([1, 1, 60, 60])
small_img=small_img.data.squeeze().numpy()#转成numpy(60,60)
plt.show()
