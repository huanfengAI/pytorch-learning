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
from torch.utils.data.dataloader import default_collate # 导入默认的拼接方式

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
		
dataset =DogCat('./dogcat/',transforms=transform)
#下面我们设置一个权重列表，狗的图片被取出的概率是猫的概率的两倍，两类图片被取出的概率与weights的绝对大小无关，只和比值有关
weights=[2 if label == 1 else 1 for data , label in dataset]
#print(weights)[2, 1, 1, 2, 1, 2, 1, 2]权重参数和样本的顺序是一一对应的
from torch.utils.data.sampler import  WeightedRandomSampler

#下面我们先来构建一个Sampler对象
sampler=WeightedRandomSampler(weights,num_samples=9,replacement=True)
#然后我们使用类加载器的时候，可以使用Sampler
dataloader=DataLoader(dataset,batch_size=3,sampler=sampler)
for datas,labels in dataloader:
    print(labels.tolist())
#[1, 1, 1]
#[0, 1, 1]
#[1, 0, 1]
#一共有8个样本，但是返回了9个肯定有重复返回的，这就是replacement的作用
