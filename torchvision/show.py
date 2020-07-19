import torchvision as tv
from torchvision import transforms as tfs
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from torch import nn
show=ToPILImage()#tensor->img

def train_tf(x):
	im_aug=tfs.Compose([
		tfs.Resize(120),#先升级到120，再裁剪96
		tfs.RandomHorizontalFlip(),
		tfs.RandomCrop(96),
		tfs.ColorJitter(brightness=0.5,contrast=0.5,hue=0.5),
		tfs.ToTensor(),
		tfs.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
	])
	x=im_aug(x)    
	return x
def test_tf(x):
	im_aug=tfs.Compose([
		tfs.Resize(96),
		tfs.ToTensor(),
		tfs.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
	])
	x=im_aug(x)
	return x

train_set = CIFAR10('./adata', train=True, transform=train_tf,download=True)
test_set = CIFAR10('./adata', train=False, transform=test_tf,download=True)
test_data =DataLoader(test_set, batch_size=128, shuffle=True)
train_data =DataLoader(train_set, batch_size=64, shuffle=True)
#显示训练集中的第101张图片
(data,label)=train_set[101]
a=show((data+1)/2).resize((100,100))#show可以把tensor转成Image方便可视化
plt.imshow(a)
plt.show()
#一口气显示一个batch-size的数据
dataiter =iter(train_data)
images,labels=dataiter.next()
b=show(tv.utils.make_grid((images+1))/2).resize((400,100))
plt.imshow(b)
plt.show()
