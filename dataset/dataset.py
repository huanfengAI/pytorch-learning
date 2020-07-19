import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np
class CatDog(Dataset):
     def __init__(self,root):
         imgs=os.listdir(root)
         self.imgs=[os.path.join(root,img)for img in imgs]
     def __getitem__(self,index):
         img_path=self.imgs[index]
         label=1 if 'dog' in img_path else 0
         pil_img=Image.open(img_path)
         array=np.asarray(pil_img)
         data=torch.from_numpy(array)
         return data,label
     def __len__(self):
         return len(self.imgs)

dataset=CatDog("dogcat")
for data,label in dataset:
    print(data.shape)
    print(label)


