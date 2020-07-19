import torch
import numpy as np
###################################################创建Tensor###############################################
#创建tensor的方式有很多：
#方式一：直接指定创建tensor的形状
#方式二：接收一个list列表
#方式三：接受一个tensor对象
#创建一个5行3列的矩阵，只是分配了空间，但是并没有进行初始化操作
y=torch.Tensor(5,3)
#print(y)
#直接接受一个list列表用户创建tensor
y12=torch.Tensor([[1,2,3],[4,5,6]])
y13=y12.tolist()#将tensor转成list类型
y14=y12.numel()#输出tensor变量y12中的元素的个数
#创建一个和y12一样大小的tensor
y15=torch.Tensor(y12.size())
#创建一个元素为1,2的tensor
y16=torch.Tensor((1,2))#tensor([1., 2.])


##############################################创建不同类型的tensor实例########################################
tensor1=torch.tensor([1,12],dtype=torch.int8)
tensor2=torch.tensor([1,12],dtype=torch.int16)
tensor3=torch.tensor([1,12],dtype=torch.int32)
tensor4=torch.tensor([1,12],dtype=torch.int64)
tensor5=torch.tensor([1,12],dtype=torch.float16)
tensor6=torch.tensor([1,12],dtype=torch.float32)
tensor7=torch.tensor([1,12],dtype=torch.float64)

############################################tensor的性质#####################################################
t2 = torch.zeros(2, 2)
#print(t2.size())#t2的size，torch.Size([2, 2])
#print(t2.dim())#t2的维度是几维，2
#print(t2.numel())#t2的元素个数,4
#############################################构造特定大小的张量##############################################
#创建一个5行3列，元素全是1的矩阵
y11=torch.ones(5,3)

# 未初始化
y37 = torch.empty(2,2) 
y37[0, 0] = 0.#对其进行初始化操作
y37[0, 1] = 1.
y37[1, 0] = 2.
y37[1, 1] = 3.
#print(y37.equal(torch.tensor([[0., 1.], [2., 3.]])))#True
#判断两个tensor是否一样
#print(y37)
#tensor([[0., 1.],
#        [2., 3.]])

y38= torch.full((2, 2, 2, 2), 3.) # 建立一个维度为（2,2,2,2）的tensor，各元素值为3.
###以上这些empty、zeros、ones、full，都有对应的torch._like形式，可以构造出一个和现有张量一样大小的张良

b=torch.zeros_like(t2)
b1=torch.ones_like(y11)
b2=torch.empty_like(y37)

#################################################维度交换和重新排列维度##########################################
#使用permute完成重拍维度
#使用transpose进行维度交换
#创建一个tensor矩阵，其中有三个维度，第一个维度大小为7，第二个维度大小为8，第三个维度大小为9
x=torch.randn(7,8,9)#torch.Size([7, 8, 9])
x=x.permute(2,0,1)#将x的1,2,3维度变为3,1,2维度torch.Size([9, 7, 8])
x=x.transpose(0,1)#将x的第一维度和第二维度调换torch.Size([7, 9, 8])

######################################################创建特殊tensor的方式########################################
#使用0,1均匀分布随机初始化二维数组,从区间[0, 1)，连续的
y1=torch.rand(5,3)
y1=torch.rand_like(torch.ones(5,3))
#print(y1)
#独立同分布生成离散均匀分布的张量，形状使用size指定，范围是low到high
y11=torch.randint(low=0,high=4,size=(3,4))
#使用randint_like函数是通过第0个参数的大小推断张量的大小的
y111=torch.randint_like(torch.ones(3,4),low=0,high=4)


#print(y11)
#创建一个5行3列，全是0的矩阵
y20=torch.zeros(5,3)
#从1到6-1，步长为2,如果不指定步长，默认步长是1
y21=torch.arange(1,6,2)#tensor([1, 3, 5])
y21=torch.arange(1,6)#tensor([1, 2, 3, 4, 5])
#从1到10切分为4份
y22=torch.linspace(1,10,4)#tensor([ 1.,  4.,  7., 10.])
#创建一个5行3列的标准正态分布,从区间[0, 1)
y23=torch.randn(5,3)
y23=torch.randn_like(torch.ones(5,3))
#print(y23)
#创建一个长度为5的随机排列
y34=torch.randperm(5)#tensor([4, 3, 2, 0, 1])
#创建对角线为1的矩阵,并不一定要是方阵
y35=torch.eye(3,3)
y36=torch.eye(3,4)#不是方阵也可以，对角线从左上角开始，直到最后一行
#构造等比数列
y37=torch.logspace(0,3,4)
#print(y37)等比数列比例为10,10的一次方到10的四次方
#离散正态分布，可以指定正态分布的均值和方差
mean=torch.tensor([0.,1.])
std=torch.tensor([3.,2.])
y38=torch.normal(mean,std)

#print(torch.mean(y38))
#print(torch.std(y38))

#torch.bernoulli(参数)函数可以生成元素值为0或1的张量，参数是一个概率张量
s=torch.full((3,4),0.6)
a=torch.bernoulli(s)
#print(a)


#############################################输出矩阵的维度#################################################
#torch.shape和torch.size()都会返回一个tuple对象的子类，所以它可以支持tuple的所有的操作，比如size()[0]
#print(y1.shape)#torch.Size([5, 3])
#print(y1.size())#torch.Size([5, 3])
#print(y1.size(0))#5
#print(y1.size()[1])#3


##########################################对tensor维度的操作###############################################
#view方法可以改变tensor的形状，要注意尺寸必须要一致，新tensor和原tensor之间内存共享
a=torch.arange(0,6)
b=a.view(-1,3)#当某一维为-1的时候，会自动计算它的大小
#如果需要添加或者减少某一维度，可以使用squeeze（减少）和unsqueeze（增加）这两个函数
#print(b.shape)#torch.Size([2, 3])
b1=b.unsqueeze(1)#在b的第一维度（维度从0开始算）增加1维#torch.Size([2, 1, 3])
b2=b1.unsqueeze(-2)#-2表示倒数第二个维度增加1维torch.Size([2, 1, 1, 3])
b3=b2.view(1,1,1,2,3)#将b2的维度修改为torch.Size([1, 1, 1, 2, 3])
b4=b3.squeeze(0)#压缩第0维中的1,torch.Size([1, 1, 2, 3])，注意只能压缩1，也就是说squeeze(3)是不可以压缩2的
b5=b3.squeeze()#压缩所有维度的1,torch.Size([2, 3])

#resize_(inplace操作)也可以调整size，如果新尺寸大于原尺寸，那么会自动分配新的内存空间，如果新尺寸小于原尺寸
b.resize_(1,3)#将b修改为维度为（1,3）
b.resize_(3,3)#将b修改为维度为（3,3）


##############################################索引操作#####################################################
c=torch.randn(3,4)#创建2个维度的矩阵，第一个维度为3，第二个维度为4，也就是一个三行四列的矩阵
#print(c[0])#获取第一维度的第0个索引
#print(c[:,0])#使用逗号进行维度分割，第一维度为所有，第二维度的索引为0
#print(c[0][2])#获取第一维度的索引是0，获取第二维度的索引是2
#print(c[0][-1])#获取第一维度的索引是0，获取第二维度的索引是倒数第一
#print(c[:2])#获取第一维度的索引是0,1
#print(c[:2,0:2])#获取第一维度的索引是0,1，第二维度的索引是0,1
#:2和0:2的区别，：2表示前2索引，而0:2表示索引0,1
#print(c[0:1,:2])#获取第一维度为0，获取第二维度为前2索引
#print(c>1)#c中的所有的元素都要和1进行比较，大于1则为1，小于1则为0，返回一个和c同样大小的矩阵

#print(c[c>1])#选出c中大于1的元素

#获取第一维度的前2索引的方式
#方式一
#print(c[:2])
#方式二
#print(c[0:2])
#方式三
#print(c[torch.LongTensor([0,1])])torch.LongTensor([0,1])表示索引，表示第一维度的0,1索引
#如果要是直接c[0,1]，那么会获取到第一维度的0索引，和第二维度的1索引，并不会获取到第一维度的前2索引

##########################################对tensor的常用函数操作gather#############################################
#gather操作，gather(input,dim,index)根据index，在dim维度上选取元素，输出的size和index一样，而index的维度也应该和input维度一样
#view(4,4)表示两个维度，第一个维度大小为4，第二个维度大小也是4
a=torch.arange(0,16).view(4,4)
#print(a)


##选取对角线上元素的两种方式
##方式一：
index=torch.LongTensor([[0,1,2,3]])
#print(index.shape)torch.Size([1, 4])
#0表示在第一维度上进行操作，因为a是二维，那么也就是在行的维度进行操作，分别获取第0行的第1列，第二行的第二列，第三行的第三列，以及第四行的第四列
b=a.gather(0,index)
#print(b)
##方式二
index1=torch.LongTensor([[0,1,2,3]]).t()#相当于转置操作
#print(index1.shape)torch.Size([4, 1])
c=a.gather(1,index1)
#print(c)

##选取反对角线上元素的两种方式
##方式一
index2=torch.LongTensor([[3,2,1,0]])
d=a.gather(0,index2)
#print(d)
##方式二
index3=torch.LongTensor([[3,2,1,0]]).t()
e=a.gather(1,index3)
#print(e)

##同时获取到正对角线和反对角线上的元素的两种方式
##方式一
index4=torch.LongTensor([[0,1,2,3],[3,2,1,0]])
f=a.gather(0,index4)
#print(f)
##方式二
index5=torch.LongTensor([[0,1,2,3],[3,2,1,0]]).t()
g=a.gather(1,index5)
#print(g)

##############################################高级索引#######################################################
##高级索引操作的结果一般不和原始的Tensor贡献内存。
x=torch.arange(0,27).view(3,3,3)
x1=x[[1, 2], [1, 2], [2, 0]] #这个相当于是x[1,1,2]和x[2,2,0]
x2=x[[2, 1, 0], [0], [1]]#x[2,0,1],x[1,0,1],x[0,0,1]
x3=x[[0, 2], ...]#这个相当于x[0] 和 x[2]
##########################################Tensor类型#######################################################
##默认的tensor是FloatTensor
t=torch.Tensor(2,3)
#print(t.dtype)torch.float32

#更改默认的Tensor类型
#torch.set_default_tensor_type('torch.DoubleTensor')
t1=torch.Tensor(2,3)
#print(t1.dtype)# 现在t1是DoubleTensor,dtype是float64
##类型转换
#方式一点上要进行转换的类型就可以了
t2=t.double()
#print(t2.dtype)torch.float64
#方式二将t1类型转变为t2类型
t3= t1.type_as(t2)
#print(t3.dtype)

#创建一个t2类型的，和t2维度一样的全是0的tensor
t4=torch.zeros_like(t2)
#print(t4.dtype)
t5=torch.zeros_like(t2,dtype=torch.int16)#还可以自己指定类型
#print(t5)
t6=torch.rand_like(t2)
#print(t6)

############################################对元素进行操作##############################################
u=torch.arange(0,6).view(2,3)
print(u)
#print(u.dtype)#torch.int64
u1=torch.pow(u,2)#对u进行指数操作
tp=torch.pow(torch.arange(1,4),torch.arange(3))
#print(torch.arange(1,4))#tensor([1, 2, 3])
#print(torch.arange(3))#tensor([0, 1, 2])
#print(tp)#tensor([1, 2, 9])1的0次方，2的一次方，3的2次方



ts = torch.sin(torch.tensor([4.0]))#4.0,三角函数操作
#print('sin = {}'.format(ts))
#print(torch.tensor([-4.0]).abs())#绝对值操作
u2=torch.mul(u,2)#对u进行2倍操作
u3=torch.clamp(u,min=3)#对u进行操作，大于3的原值，小于3的为3
#print(u.float().sqrt())#开平方操作，需要转成float类型
#print(u.float().reciprocal())#求倒数操作
#print(u.float().rsqrt())#先开方在取倒数
#print(u.sum())#求和
#print(u.prod())#乘积
#print(u.cumsum(dim=0))#在0维度上进行累加
#print(u.cumprod(dim=1))#在1维度上进行累乘
#print(u.float().norm(2))#求范数

###############################################张量的点积##############################################
##dot
##一维张量和一维张量之间的点积，可以理解为矩阵中的向量之间的点积操作
xx=torch.arange(5)
#print(xx)tensor([0, 1, 2, 3, 4])
yy=torch.arange(1,6)
#print(yy)#tensor([1, 2, 3, 4, 5])
#print(torch.dot(xx,yy))tensor(40)=0*1+1*2+2*3+3*4+4*5=2+6+12+20=40


##二维张量和一维张量之间的点积
xx=torch.arange(4).view(2,2)
#print(xx)
#tensor([[0, 1],
#        [2, 3]])
yy=torch.arange(2)
#print(yy)#tensor([0, 1])
#print(torch.mv(xx,yy))#tensor([1, 3])1=0*0+1*1,3=2*0+3*1


##二维张量和二维张量之间的张量点积,就是矩阵中的乘积操作
xx=torch.arange(6).view(2,3)
#print(xx)
#tensor([[0, 1, 2],
#        [3, 4, 5]])

yy=torch.arange(6).view(3,2)
#print(yy)
#tensor([[0, 1],
#        [2, 3],
#        [4, 5]])
#print(torch.mm(xx,yy))
#tensor([[10, 13],
#        [28, 40]])
##############################################对tensor进行归并操作######################################
#此类操作会使输出形状小于输入形状，并可以沿着某一维度进行指定操作。参数dim，用来指定这些操作是在哪个维度上执行的
b=torch.ones(2,3)
#print(b.sum(dim = 0, keepdim=True))#keepdim=True会保留维度1,tensor([[2., 2., 2.]])
#print(b.sum(dim=0,keepdim=False))#tensor([2., 2., 2.])
#print(b.sum(dim=1))#tensor([3., 3.])
bb = torch.arange(0, 6).view(2, 3)
#tensor([[ 0.,  1.,  2.],
#       [ 3.,  4.,  5.]])
#print(bb.cumsum(dim=1)) # 沿着行累加
#tensor([[  0.,   1.,   3.],
#        [  3.,   7.,  12.]])



###########################################选取部分张量元素#############################################
##tensor.index_select(维度，索引)
t=torch.arange(24).reshape(2,3,4)#创建一个有三个维度的tensor
index=torch.tensor([1,2])#
#print(t.index_select(1,index))#在第二个维度上进行操作，索引为1,2

##一个维度的切片操作
t=torch.arange(12)
#print(t)tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
#print(t[3])tensor(3)
#print(t[-5])tensor(7)
#print(t[3:6])tensor([3, 4, 5])
#print(t[:6])tensor([0, 1, 2, 3, 4, 5])
#print(t[3:])tensor([ 3,  4,  5,  6,  7,  8,  9, 10, 11])
#print(t[-5:])tensor([ 7,  8,  9, 10, 11])
#print(t[3:6:2])tensor([3, 5])取3到6之间的，间隔为2取
#print(t[3::2])tensor([ 3,  5,  7,  9, 11])，取3之后的间隔为2取

##多个维度的切片操作,不同维度之间的操作使用逗号隔开
t=torch.arange(12).reshape(3,4)
#print(t)
#print(t[2:,-2])#第一维度为索引2之后的，第二维度为倒数第二个，tensor([10])
#print(t[0,:])#第一维度索引0的，第二维度全部，tensor([0, 1, 2, 3])


#masked_select为选择一个tensor中指定的元素，参数必须和调用这个方法的tensor维度一致，且必须是uint8，
t=torch.arange(12).reshape(3,4)
#print(t)
mask=torch.tensor([[1,0,0,1],[0,1,1,0],[0,0,1,0]],dtype=torch.uint8)
#print(t.masked_select(mask))tensor([ 0,  3,  5,  6, 10])

#mask中为1的，t中就留下了。一维张量的元素的个数就是张量mask中1的个数

#take()和masked_select不同之处在于，我们传递的参数不用和tensor维度一致了，take()函数将张量各元素按照唯一的指标进行索引，相当于对经过reshape(-1)操作后的张量进行索引
t=torch.arange(12).reshape(3,4)
index=torch.tensor([3,5,6])
#print(t.take(index))#tensor([3, 5, 6])

##nonzero(input)获取到input中非0元素的下标
#print(torch.nonzero(t))

########################################对tensor中的元素进行元素比较，判断它是否符合条件##################################
z1=torch.linspace(0,15,6).view(2,3)
z2=torch.linspace(15,0,6).view(2,3)
#print(z1)
#print(z2)
#print(z1>z2)#z1和z2逐元素进行比较，符合条件返回1，不符合条件返回0
#print(z1[z1>z2])#返回z1中大于z2的元素
#print(torch.max(z1))返回z1中最大的元素
#print(torch.max(z1,dim=1))返回在第二维度的最大值
#print(torch.max(z1,dim=1))#(tensor([ 6., 15.]), tensor([2, 2])),后面的2，2表示6的下标索引在第一行为2,15的下标索引在第二行为2
#print(torch.max(z1,z2))逐一比较两个tensor最大值,留下最大值,构成一个新的矩阵
#print(torch.min(z1,z2))逐一比较两个tensor最大值,留下最小值，构成一个新的矩阵


###########################################对tensor进行常见的线性代数的操作############################################
a = torch.linspace(0, 15, 6).view(2, 3)
#print(a.diag())#diag为对角线元素
#print(a.t())#转置操作
aa=a.t()
#矩阵的转置会导致存储空间不连续
#print(aa.is_contiguous())#False。需调用它的.contiguous方法将其转为连续。
#print(aa.contiguous())
#############################################张量的拓展和拼接#########################################################
##repeat()可以将张量的内容进行重复，使得张量的大小变大。
k1=torch.tensor([[5.,-9.]])
#print(k1.size())#torch.Size([1, 2])
k2=k1.repeat(3,3)#torch.Size([3, 3])
#print(k2.size())#torch.Size([3,6])
#repeat之后新向量的大小为1*3=3,2*3=6

##cat拼接
#要想完成拼接，需要符合以下条件：
#拼接的张量应该有相同的维度，比如一个张量有四维，那么另外一个张量也应该有4维
#如果在第n维进行拼接，那么张量在其他维度的大小应该是一样的，拼接出来的新的张量在第n维是各张量第n维之和
tp=torch.arange(12).reshape(3,4)
tn=-tp
tc0=torch.cat([tp,tn],0)#torch.Size([6, 4])
#print(tc0)
#tensor([[  0,   1,   2,   3],
#        [  4,   5,   6,   7],
#        [  8,   9,  10,  11],
#        [  0,  -1,  -2,  -3],
#        [ -4,  -5,  -6,  -7],
#        [ -8,  -9, -10, -11]])

tc1=torch.cat([tp,tp,tn,tn],1)##torch.Size([3, 16])
#print(tc1.shape)

##stack拼接
#stack拼接和cat拼接是不一样的，首先stack要想拼接需要需要输入张量的大小完全一样，拼接完成的新张量会比输入张量的维度多一个维度
#多出来的维度就是新拼接出来的维度，那个维度的大小就是输入张量的个数
tp=torch.arange(12).reshape(3,4)
tn=-tp
ts0=torch.stack([tp,tn],0)#torch.Size([2, 3, 4])
#print(ts0)
#tensor([[[  0,   1,   2,   3],
#         [  4,   5,   6,   7],
#         [  8,   9,  10,  11]],

#        [[  0,  -1,  -2,  -3],
#         [ -4,  -5,  -6,  -7],
#         [ -8,  -9, -10, -11]]])

ts1=torch.stack([tp,tp,tn,tn],1)#torch.Size([3, 4, 4])
#print(ts1.size())

#############################################pytorch的广播机制##################################################
##并不是任意两个张量之间都可以进行广播操作，首先两个张量的维度必须都大于等于1
##然后从后往前比较两个张量的大小，张量的大小必须相同，如果不相同只能符合两种情况
##情况一：其中一个张量的大小没有条目了
##情况二：其中一个张量的维度为1
##比如（2,3,4）和（2,1,4）是可以的，符合情况二
##比如（2,3,4）和（1,1,4）是可以的，符合情况二
##比如（2,3,4）和（4）是可以的，符合情况一
##比如（2,3,4）和（2,4）是不可以的，从后看，第一个是3，第二个是2，这样既不符合情况一，也不符合情况二
r = torch.ones(3, 2)
r1 = torch.zeros(2, 3,1)
#手动实行广播机制，方式一
#print(r.view(1,3,2).expand(2,3,2)+r1.expand(2,3,2))
#手动实行广播机制，方式二
#print(r[None].expand(2, 3, 2) + r1.expand(2,3,2))
###########################################加法运算的四种方式###################################################
#1
y2=y+y1
#2#普通加法不改变y的内容
y3=y.add(y1)
#3
y4=torch.add(y,y1)
#4out表示将相加的结果赋值给y5，那么此时就可以直接输出y5，就可以获取到相加的值
y5=torch.Tensor(5,3)
torch.add(y,y1,out=y5)
#print(y5)

######################################numpy和tensor之间的转换操作###############################################
#numpy转Tensor会转成float类型，如果numpy的类型不是float类型，那么内存不会共享
#Tensor和numpy之间内存共享，也就是说如果其中一个变了，另外一个也会发生改变
#Tensor转numpy只需要tensor对象直接.numpy就ok
y6=torch.ones(5,3)
y7=y6.numpy()#y7是numpy类型
#print(y7)

#numpy转tensor需要使用torch.from_numpy(numpy对象)
n=np.ones((5,3))#numpy中传递不能直接5,3而是应该使用一个tuple元祖来完成((5,3))
t=torch.from_numpy(n)#方式一：将numpy转成了tensor类型
t1=torch.Tensor(n)#方式二：将numpy转成了tensor类型，只进行数据拷贝，不会共享内存

#将tensor转成可以在Gpu的tensor
if torch.cuda.is_available():
	yy=y1.cuda()#这就将y1转变为了可以在Gpu上运行的yy
	print(yy)

device = torch.device('cpu')
a.to(device)
##如果神经网络要在Gpu上训练，我们需要进行如下操作
#device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
#net.to(device)#神经网络也要转到Gpu上
#images = images.to(device)
#labels = labels.to(device)
#output = net(images)

#gpu_tensor = torch.randn(10, 20).cuda(0) # 将 tensor 放到第一个 GPU 上
#gpu_tensor = torch.randn(10, 20).cuda(1) # 将 tensor 放到第二个 GPU 上

#dtype = torch.cuda.FloatTensor # 定义默认 GPU 的 数据类型
#gpu_tensor = torch.randn(10, 20).type(dtype)
############################################持久化##################################################
#Tensor的保存和加载十分的简单，使用t.save和t.load即可完成相应的功能。

if t.cuda.is_available():
    a = a.cuda(1) # 把a转为GPU1上的tensor,
    t.save(a,'a.pth')

    # 加载为b, 存储于GPU1上(因为保存时tensor就在GPU1上)
    b = t.load('a.pth')
    # 加载为c, 存储于CPU
    c = t.load('a.pth', map_location=lambda storage, loc: storage)
    # 加载为d, 存储于GPU0上
    d = t.load('a.pth', map_location={'cuda:1':'cuda:0'})
