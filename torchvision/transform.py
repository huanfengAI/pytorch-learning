from PIL import Image
from torchvision import transforms as tfs
im=Image.open('./cat.png')
print('before scale,shape{}'.format(im.size))
new_im=tfs.Resize((100,200))(im)#将图片缩放到（100,200）
new_im=tfs.Resize(100)(im)#将图片最短边缩放到100，长边等比例变化
new_im=tfs.RandomCrop(100)(im)#随机裁剪出（100,100）的图片
new_im=tfs.RandomCrop((100,100))(im)#随机裁剪出（100,150）的图片
new_im=tfs.CenterCrop(100)(im)#中心裁剪出（100,100）的图片
new_im=tfs.CenterCrop((100,150))(im)#中心裁剪出（100,150）的图片
new_im=tfs.RandomHorizontalFlip()(im)#水平翻转
new_im=tfs.RandomVerticalFlip()(im)#垂直翻转
new_im=tfs.RandomRotation(45)(im)#随机-45到45之间旋转
new_im=tfs.ColorJitter(brightness=1)(im)## 随机从 0 ~ 2 之间亮度变化，1 表示原图
new_im=tfs.ColorJitter(contrast=1)(im)# 随机从 0 ~ 2 之间对比度变化，1 表示原图
new_im=tfs.ColorJitter(hue=0.5)(im)#随机从-0.5~0.5之间对颜色变化
im_aug =tfs.Compose([#使用Compose将这些数据增强给组合起来
	tfs.Resize(120),#组合起来的时候。图片应该放在第一个位置
	tfs.RandomHorizontalFlip(),
	tfs.RandomCrop(96),
	tfs.ColorJitter(brightness=0.5,contrast=0.5,hue=0.5)
])

new_im=im_aug(im)#使用整合到一起的数据增强组合来对图片进行处理

