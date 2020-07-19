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
	
def my_collate_fn(batch):#batch表示一个batch中的所有的数据
	batch =list(filter(lambda x:x[0] is not None ,batch))#过滤掉有问题的样本，这里有问题的样本我们上面已经处理返回None
	if len(batch)==0:return torch.tensor()
	return default_collate(batch)

             
class NewDogCat(DogCat): #继承前面实现的DogCat数据集
	def __getitem__(self,index):
		try:
			return super(NewDogCat,self).__getitem__(index)
		except:
                        new_index=random.randint(0,len(self)-1)
                        return self[new_index]

dataset=NewDogCat('./dogcat_w/',transforms=transform)
dataloader =DataLoader(dataset,batch_size=2,collate_fn=my_collate_fn,num_workers=1,shuffle=True)
for data in dataloader:
    print(data[0].shape)
    print(data[1])
