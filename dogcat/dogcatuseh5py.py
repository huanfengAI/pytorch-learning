import torch
from torch.utils.data import Dataset
import h5py

data_transforms = {
    'train':
    transforms.Compose([
        transforms.RandomSizedCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'val':
    transforms.Compose([
        transforms.Scale(320),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}


class classfier(nn.Module):
	def __init__(self,dim,n_classes):
		super(classfier,self).__init__()
		self.fc=nn.Sequential(
			nn.Linear(dim,1000),
			nn.ReLU(True),
			nn.Dropout(0.5),
			nn.Linear(1000,n_classes)
		)
	def forward(x):
		x=self.fc(x)
		return x





class MyDataset(Dataset):
	def __init__(self,h5file_list):
		label=h5py.File(h5file_list[0],'r')#读取一个h5文件,任何一个都行，因为标签都是一样的
		self.label=torch.from_numpy(label['label'].value)#获取到标签,当时存到这里面是numpy，所以这里需要转成tensor
		self.nSamples=self.label.size(0)#获取到样本的数量
		temp_dataset=torch.FloatTensor()
		for file in h5file_list:#读取每一个文件
			h5_file =h5py.File(file,'r')
			dataset=torch.from_numpy(h5_file['data'].value)
			temp_dataset=torch.cat((temp_dataset,dataset),1)
		
		self.dataset=temp_dataset
	def __len__(self):
		return self.nSamples
	def __getitem__(self,index):
		data=self.dataset[index]
		label=self.label[index]
		
		return (data,label)
		
#创建train和test文件列表
train_list=['train_data_feature_{}.hd5f'.format(i) for i in ['vgg', 'inceptionv3', 'resnet152']]
val_list=['test_data_feature_{}.hd5f'.format(i) for i in ['vgg', 'inceptionv3', 'resnet152']]

#创建trian和test的dataloader
train_data=DataLoader(MyDataset['train_list'],batch_size=2,shuffle=False,num_workers=2,transform=data_transforms['train'])
test_data=DataLoader(MyDataset['val_list'],batch_size=2,shuffle=True,num_works=2,transform=data_transforms['test'])


input=MyDataset['train_list'].dataset.size(1)

net=classfier(input,2)
criterion =nn.CrossEntropyLoss()#定义损失函数
optimizer =torch.optim.SGD(net.parameters(),lr=1e-3)
#训练
prev_time=datetime.now()
for epoch in range(30):
	train_loss=0
	train_acc =0
	net =net.train()
	for im ,label in train_data:#im,label为一批数据，也就是64个样本
		#前向传播并计算损失
		#print(im.size())#im=im.view(im.size(0),-1)torch.Size([64, 1, 28, 28])
		#im=im.view(im.size(0),-1)
		#print(im.size())torch.Size([64, 784])
		output =net(im)
		
		loss =criterion(output ,label)
		#反向传播
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		#print(loss.data)
		train_loss +=loss.data.float()
		train_acc +=get_acc(output,label)
		print(train_acc)
		#print(train_acc/len(train_data))
		#print(train_acc/64)
	#测试
	cur_time =datetime.now()
	h,remainder =divmod((cur_time-prev_time).seconds,3600)
	m,s=divmod(remainder,60)
	time_str ="Time %02d:%02d:%02d"%(h,m,s)
	valid_loss=0
	valid_acc=0
	net =net.eval()
	for im,label in test_data:
		#im=im.view(im.size(0),-1)
		output =net(im)
		loss= criterion(output,label)
		valid_loss +=loss.data.float()
		valid_acc +=get_acc(output,label)
	epoch_str=(
			"Epoch %d. Train Loss %f,Train Acc:%f,Valid Loss: %f,Valid Acc: %f ,"
			%(epoch,train_loss/len(train_data),
			  train_acc /len(train_data),
			  valid_loss/len(test_data),
			  valid_acc /len(test_data)))
	prev_time=cur_time
	print(epoch_str+time_str)#训练一批测试一批,time_str为每次epoch运行的时间00:00:07表示7秒
