import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np
from torchvision import datasets,transforms
transform=transforms.Compose([
	transforms.Resize(224),
	transforms.CenterCrop(220),
	transforms.ToTensor(),
	transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])

])
class CatDog(Dataset):
     def __init__(self,root,transforms):
         self.transforms=transforms
         imgs=os.listdir(root)
         self.imgs=[os.path.join(root,img)for img in imgs]
     def __getitem__(self,index):
         img_path=self.imgs[index]
         label=1 if 'dog' in img_path else 0
         data=Image.open(img_path)
         data=self.transforms(data)
         return data,label
     def __len__(self):
         return len(self.imgs)

dataset=CatDog("dogcat",transform)
for data,label in dataset:
    print(data.shape)
    print(label)


