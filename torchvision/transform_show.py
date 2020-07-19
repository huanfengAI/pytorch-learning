import torch
from PIL import Image
from torchvision import transforms as tfs
import matplotlib.pyplot as plt
from torchvision import datasets
im=Image.open('./cat.png')
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
