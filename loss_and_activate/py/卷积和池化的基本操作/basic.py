from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from torch import nn
im=Image.open('./cat.png').convert('L')#转成灰度图
im=np.array(im,dtype='float32')#图片转numpy
#plt.imshow(im.astype('uint8'),cmap='gray')
im=torch.from_numpy(im.reshape((1,1,im.shape[0],im.shape[1])))#numpy转tensor，并且图片的shape变为了（1,1，im.shape[0],im.shape[1])，也就是(1,1,长，宽)

#定义一个卷积核，bias=False表示不适用bias，这个卷积和的输入通道是1，输出通道是1，卷积核大小为3*3
conv1=nn.Conv2d(1,1,3,bias=False)
#定义卷积核参数
sobel_kernel=np.array([[-1,-1,-1],[-1,-8,-1],[-1,-1,-1]],dtype='float32')
#更改卷积核的维度，因为卷积层的卷积为1,1，3*3，所以我们要将卷积核的维度也弄成和conv1一样的
sobel_kernel=sobel_kernel.reshape((1,1,3,3))
#对卷积层的权重进行赋值
conv1.weight.data=torch.from_numpy(sobel_kernel)
#对图片进行卷积操作
edge1=conv1(im)

edge1=edge1.data.squeeze().numpy()#将输出格式转为图片格式（numpy），转之前先将batch-size的1，和通道的1去掉
print(edge1.shape[2])
plt.imshow(edge1,cmap='gray')
#定义一个最大池化，大小为2,2
pool1=nn.MaxPool2d(2,2)
print('before max pool, image shape: {} x {}'.format(im.shape[2], im.shape[3]))
#对图片进行最大池化操作
small_img=pool1(im)
small_im1 =small_img.data.squeeze().numpy()
print('after max pool, image shape: {} x {} '.format(small_im1.shape[0], small_im1.shape[1]))
plt.show()#显示
