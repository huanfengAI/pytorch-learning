import torch
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader
import os
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import random
from torchvision import datasets
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid,save_image

transform =transforms.Compose([
	transforms.Resize(224),
	transforms.CenterCrop(224),
	transforms.ToTensor(),#将图片image转成tensor，他会自动归一化到[0,1]
	transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])#标准化至[-1,1],规定均值和标准差

])

class DogCat(data.Dataset):
	def __init__(self,root,transforms=None):
		imgs=os.listdir(root)
		self.imgs=[os.path.join(root,img) for img in imgs]
		self.transforms=transforms
	
	def __getitem__(self,index):
		img_path=self.imgs[index]
		label= 0 if 'dog' in img_path.split('/')[-1] else 1
		data =Image.open(img_path)
		if self.transforms:
			data=self.transforms(data)
		return data,label
	def __len__(self):
		return len(self.imgs)
#torchvison的两个常用的函数，
#make_grid,它能够将多张图片拼接成一个网格中
#save_img它能将tensor保存成图片
dataset =DogCat('./dogcat/',transforms=transform)

dataloader =DataLoader(dataset,shuffle=True,batch_size=16)

dataiter =iter(dataloader)
#print(len(next(dataiter)[0]))8,因为一共有8张图片，所以虽然batch_size为16，但是只能取出8张来
img =make_grid(next(dataiter)[0],4)#拼接成4*4的网格图片，并且会将图片转成3通道的
#print(img)<PIL.Image.Image image mode=RGB size=906x454 at 0x7F102350C240>
a=ToPILImage(img)#将tensor转成图片
#print(a)#<PIL.Image.Image image mode=RGB size=906x454 at 0x7F2C637872B0>
save_image(img, 'a.png')#保存图片，名称是a.png
#Image.open('a.png')
