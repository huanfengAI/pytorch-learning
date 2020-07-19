# ### 5.4 使用GPU加速：cuda
# 这部分内容在前面介绍Tensor、Module时大都提到过，这里将做一个总结，并深入介绍相关应用。
# 
# 在PyTorch中以下数据结构分为CPU和GPU两个版本：
# - Tensor
# - nn.Module（包括常用的layer、loss function，以及容器Sequential等）
# 
# 它们都带有一个`.cuda`方法，调用此方法即可将其转为对应的GPU对象。注意，`tensor.cuda`会返回一个新对象，这个新对象的数据已转移至GPU，而之前的tensor还在原来的设备上（CPU）。而`module.cuda`则会将所有的数据都迁移至GPU，并返回自己。所以`module = module.cuda()`和`module.cuda()`所起的作用一致。
# 
# nn.Module在GPU与CPU之间的转换，本质上还是利用了Tensor在GPU和CPU之间的转换。`nn.Module`的cuda方法是将nn.Module下的所有parameter（包括子module的parameter）都转移至GPU，而Parameter本质上也是tensor(Tensor的子类)。
# 下面将举例说明，这部分代码需要你具有两块GPU设备。
# 
# P.S. 为什么将数据转移至GPU的方法叫做`.cuda`而不是`.gpu`，就像将数据转移至CPU调用的方法是`.cpu`？这是因为GPU的编程接口采用CUDA，而目前并不是所有的GPU都支持CUDA，只有部分Nvidia的GPU才支持。PyTorch未来可能会支持AMD的GPU，而AMD GPU的编程接口采用OpenCL，因此PyTorch还预留着`.cl`方法，用于以后支持AMD等的GPU。


#tensor = t.Tensor(3, 4)
# 返回一个新的tensor，保存在第1块GPU上，但原来的tensor并没有改变
#tensor.cuda(0)
#tensor.is_cuda # False


# In[40]:


# 不指定所使用的GPU设备，将默认使用第1块GPU
#tensor = tensor.cuda()
#tensor.is_cuda # True


# In[43]:


#module = nn.Linear(3, 4)
#module.cuda(device = 1)
#module.weight.is_cuda # True


# In[44]:


#class VeryBigModule(nn.Module):
#    def __init__(self):
#        super(VeryBigModule, self).__init__()
##        self.GiantParameter1 = t.nn.Parameter(t.randn(100000, 20000)).cuda(0)
#        self.GiantParameter2 = t.nn.Parameter(t.randn(20000, 100000)).cuda(1)
    
#    def forward(self, x):
#        x = self.GiantParameter1.mm(x.cuda(0))
#        x = self.GiantParameter2.mm(x.cuda(1))
#        return x

# 
# 上面最后一部分中，两个Parameter所占用的内存空间都非常大，大概是8个G，如果将这两个都同时放在一块GPU上几乎会将显存占满，无法再进行任何其它运算。此时可通过这种方式将不同的计算分布到不同的GPU中。

# 关于使用GPU的一些建议：
# - GPU运算很快，但对于很小的运算量来说，并不能体现出它的优势，因此对于一些简单的操作可直接利用CPU完成
# - 数据在CPU和GPU之间，以及GPU与GPU之间的传递会比较耗时，应当尽量避免
# - 在进行低精度的计算时，可以考虑`HalfTensor`，它相比于`FloatTensor`能节省一半的显存，但需千万注意数值溢出的情况。

# 另外这里需要专门提一下，大部分的损失函数也都属于`nn.Moudle`，但在使用GPU时，很多时候我们都忘记使用它的`.cuda`方法，这在大多数情况下不会报错，因为损失函数本身没有可学习的参数（learnable parameters）。但在某些情况下会出现问题，为了保险起见同时也为了代码更规范，应记得调用`criterion.cuda`。下面举例说明。

# In[45]:


# 交叉熵损失函数，带权重
criterion = t.nn.CrossEntropyLoss(weight=t.Tensor([1, 3]))
input = t.randn(4, 2).cuda()
target = t.Tensor([1, 0, 0, 1]).long().cuda()

# 下面这行会报错，因weight未被转移至GPU
# loss = criterion(input, target)

# 这行则不会报错
criterion.cuda()
loss = criterion(input, target)

criterion._buffers


# 而除了调用对象的`.cuda`方法之外，还可以使用`torch.cuda.device`，来指定默认使用哪一块GPU，或使用`torch.set_default_tensor_type`使程序默认使用GPU，不需要手动调用cuda。

# In[46]:


# 如果未指定使用哪块GPU，默认使用GPU 0
x = t.cuda.FloatTensor(2, 3)
# x.get_device() == 0
y = t.FloatTensor(2, 3).cuda()
# y.get_device() == 0

# 指定默认使用GPU 1
with t.cuda.device(1):    
    # 在GPU 1上构建tensor
    a = t.cuda.FloatTensor(2, 3)

    # 将tensor转移至GPU 1
    b = t.FloatTensor(2, 3).cuda()
    print(a.get_device() == b.get_device() == 1 )

    c = a + b
    print(c.get_device() == 1)

    z = x + y
    print(z.get_device() == 0)

    # 手动指定使用GPU 0
    d = t.randn(2, 3).cuda(0)
    print(d.get_device() == 2)


# In[47]:


t.set_default_tensor_type('torch.cuda.FloatTensor') # 指定默认tensor的类型为GPU上的FloatTensor
a = t.ones(2, 3)
a.is_cuda


# 如果服务器具有多个GPU，`tensor.cuda()`方法会将tensor保存到第一块GPU上，等价于`tensor.cuda(0)`。此时如果想使用第二块GPU，需手动指定`tensor.cuda(1)`，而这需要修改大量代码，很是繁琐。这里有两种替代方法：
# 
# - 一种是先调用`t.cuda.set_device(1)`指定使用第二块GPU，后续的`.cuda()`都无需更改，切换GPU只需修改这一行代码。
# - 更推荐的方法是设置环境变量`CUDA_VISIBLE_DEVICES`，例如当`export CUDA_VISIBLE_DEVICE=1`（下标是从0开始，1代表第二块GPU），只使用第二块物理GPU，但在程序中这块GPU会被看成是第一块逻辑GPU，因此此时调用`tensor.cuda()`会将Tensor转移至第二块物理GPU。`CUDA_VISIBLE_DEVICES`还可以指定多个GPU，如`export CUDA_VISIBLE_DEVICES=0,2,3`，那么第一、三、四块物理GPU会被映射成第一、二、三块逻辑GPU，`tensor.cuda(1)`会将Tensor转移到第三块物理GPU上。
# 
# 设置`CUDA_VISIBLE_DEVICES`有两种方法，一种是在命令行中`CUDA_VISIBLE_DEVICES=0,1 python main.py`，一种是在程序中`import os;os.environ["CUDA_VISIBLE_DEVICES"] = "2"`。如果使用IPython或者Jupyter notebook，还可以使用`%env CUDA_VISIBLE_DEVICES=1,2`来设置环境变量。

# 从 0.4 版本开始，pytorch新增了`tensor.to(device)`方法，能够实现设备透明，便于实现CPU/GPU兼容。这部份内容已经在第三章讲解过了。

# 从PyTorch 0.2版本中，PyTorch新增分布式GPU支持。分布式是指有多个GPU在多台服务器上，而并行一般指的是一台服务器上的多个GPU。分布式涉及到了服务器之间的通信，因此比较复杂，PyTorch封装了相应的接口，可以用几句简单的代码实现分布式训练。分布式对普通用户来说比较遥远，因为搭建一个分布式集群的代价十分大，使用也比较复杂。相比之下一机多卡更加现实。对于分布式训练，这里不做太多的介绍，感兴趣的读者可参考文档[^distributed]。
# [^distributed]: http://pytorch.org/docs/distributed.html

# #### 5.4.1 单机多卡并行
# 要实现模型单机多卡十分容易，直接使用 `new_module = nn.DataParallel(module, device_ids)`, 默认会把模型分布到所有的卡上。多卡并行的机制如下：
# - 将模型（module）复制到每一张卡上
# - 将形状为（N,C,H,W）的输入均等分为 n份（假设有n张卡），每一份形状是（N/n, C,H,W）,然后在每张卡前向传播，反向传播，梯度求平均。要求batch-size 大于等于卡的个数(N>=n)
# 
# 在绝大多数情况下，new_module的用法和module一致，除了极其特殊的情况下（RNN中的PackedSequence）。另外想要获取原始的单卡模型，需要通过`new_module.module`访问。


# ### 5.5  持久化
# 在PyTorch中，以下对象可以持久化到硬盘，并能通过相应的方法加载到内存中：
# - Tensor
# - Variable
# - nn.Module
# - Optimizer
# 
# 本质上上述这些信息最终都是保存成Tensor。Tensor的保存和加载十分的简单，使用t.save和t.load即可完成相应的功能。在save/load时可指定使用的pickle模块，在load时还可将GPU tensor映射到CPU或其它GPU上。
# 
# 我们可以通过`t.save(obj, file_name)`等方法保存任意可序列化的对象，然后通过`obj = t.load(file_name)`方法加载保存的数据。对于Module和Optimizer对象，这里建议保存对应的`state_dict`，而不是直接保存整个Module/Optimizer对象。Optimizer对象保存的主要是参数，以及动量信息，通过加载之前的动量信息，能够有效地减少模型震荡，下面举例说明。
# 

# In[39]:


a = torch.Tensor(3, 4)
if torch.cuda.is_available():
        a = a.cuda(1) # 把a转为GPU1上的tensor,
        torch.save(a,'a.pth')
        
        # 加载为b, 存储于GPU1上(因为保存时tensor就在GPU1上)
        b = torch.load('a.pth')
        
        # 加载为c, 存储于CPU
        c = torch.load('a.pth', map_location=lambda storage, loc: storage)
        
        # 加载为d, 存储于GPU0上
        d = torch.load('a.pth', map_location={'cuda:1':'cuda:0'})


# In[40]:


torch.set_default_tensor_type('torch.FloatTensor')
from torchvision.models import SqueezeNet
model = SqueezeNet()
# module的state_dict是一个字典
model.state_dict().keys()


# Module对象的保存与加载
torch.save(model.state_dict(), 'squeezenet.pth')
model.load_state_dict(torch.load('squeezenet.pth'))



optimizer = torch.optim.Adam(model.parameters(), lr=0.1)


torch.save(optimizer.state_dict(), 'optimizer.pth')
optimizer.load_state_dict(torch.load('optimizer.pth'))

all_data = dict(
    optimizer = optimizer.state_dict(),
    model = model.state_dict(),
    info = u'模型和优化器的所有参数'
)
torch.save(all_data, 'all.pth')

all_data = torch.load('all.pth')
all_data.keys()
