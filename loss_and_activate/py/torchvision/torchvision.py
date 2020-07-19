import torch
from PIL import Image
from torchvision import transforms as tfs
import matplotlib.pyplot as plt
#计算机视觉工具包torchvision
#torchvision主要包含三个方面
#第一个方面是models：提供深度学习中各种经典网络的网络结构以及预训练好的模型，包括`AlexNet`、VGG系列、ResNet系列、Inception系列等。
# - datasets： 提供常用的数据集加载，设计上都是继承`torhc.utils.data.Dataset`，主要包括`MNIST`、`CIFAR10/100`、`ImageNet`、`COCO`等。
# - transforms：提供常用的数据预处理操作，主要包括对Tensor以及PIL Image对象的操作。
#我们下面来看一下如何才能加载已经训练好的模型
from torchvision import models
#resnet34=models.squeezenet1_1(pretrained=True,num_classes=1000)
#resnet34.fc=nn.Linear(512,10)#然后修改其全连接层，使其变为10分类的问题
#模型下载的位置为/home/dongxinge/.torch/models/squeezenet1_1-f364aa15.pth

#数据集
from torchvision import datasets
#dataset=datasets.MNIST('data/',download=True,train=False,transform=transforms)
#data/为数据集下载的位置
#train=False为下载的数据为测试集


#transform为下载的数据进行转换操作

from torchvision import transforms
to_pil =transforms.ToPILImage()#构建一个transform转换操作
a=to_pil(torch.randn(3,64,64))#对torch.randn(3,64,64)进行转换操作。此时它还是图片，并不是张量


#torchvison的两个常用的函数，
#make_grid,它能够将多张图片拼接成一个网格中
#save_img它能将tensor保存成图片

dataloader =DataLoader(dataset,shuffle=True,batch_size=16)
from torchvision.utils import make_grid,save_image
dataiter =iter(dataloader)
#print(len(next(dataiter)[0]))8,因为一共有8张图片，所以虽然batch_size为16，但是只能取出8张来
img =make_grid(next(dataiter)[0],4)#拼接成4*4的网格图片，并且会将图片转成3通道的
#print(img)<PIL.Image.Image image mode=RGB size=906x454 at 0x7F102350C240>
a=to_img(img)#将tensor转成图片
#print(a)#<PIL.Image.Image image mode=RGB size=906x454 at 0x7F2C637872B0>
#save_image(img, 'a.png')#保存图片，名称是a.png
#Image.open('a.png')

im=Image.open('./cat.png')
print('before scale,shape{}'.format(im.size))
#new_im=tfs.Resize((100,200))(im)#将图片缩放到（100,200）
#new_im=tfs.Resize(100)(im)将图片最短边缩放到100，长边等比例变化
#new_im=tfs.RandomCrop(100)(im)#随机裁剪出（100,100）的图片
#new_im=tfs.RandomCrop((100,150))(im)#随机裁剪出（100,150）的图片
#new_im=tfs.CenterCrop(100)(im)#中心裁剪出（100,100）的图片
#new_im=tfs.CenterCrop((100,150))(im)#中心裁剪出（100,150）的图片
#new_im=tfs.RandomHorizontalFlip()(im)#水平翻转
#new_im=tfs.RandomVerticalFlip()(im)#垂直翻转
#new_im=tfs.RandomRotation(45)(im)#随机-45到45之间旋转
#new_im=tfs.ColorJitter(brightness=1)(im)## 随机从 0 ~ 2 之间亮度变化，1 表示原图
#new_im=tfs.ColorJitter(contrast=1)(im)# 随机从 0 ~ 2 之间对比度变化，1 表示原图
#new_im=tfs.ColorJitter(hue=0.5)(im)#随机从-0.5~0.5之间对颜色变化
im_aug =tfs.Compose([#使用Compose将这些数据增强给组合起来
	tfs.Resize(120),#组合起来的时候。图片应该放在第一个位置
	tfs.RandomHorizontalFlip(),
	tfs.RandomCrop(96),
	tfs.ColorJitter(brightness=0.5,contrast=0.5,hue=0.5)
])
#new_im=im_aug(im)#使用整合到一起的数据增强组合来对图片进行处理
nrows=3
ncols=3
figsize=(8,8)#画布大小为8*8分为9块，行为3块，列为3块
_,figs =plt.subplots(nrows,ncols,figsize=figsize)
for i in range(nrows):
	for j in range(ncols):
		figs[i][j].imshow(im_aug(im))
		figs[i][j].axes.get_xaxis().set_visible(False)
		figs[i][j].axes.get_yaxis().set_visible(False)

#print('after scale ,shape{}'.format(new_im.size))

#plt.imshow(new_im,cmap='gray')
plt.show()
