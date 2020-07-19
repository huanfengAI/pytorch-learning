import torch
from torch.autograd import Variable



########################################################
x= torch.randn(2,3) #创建一个随机矩阵，1行3列，类型是tensor
print(x)
x=Variable(x, requires_grad=True)
print(x)
y=x*3
y.backward(torch.FloatTensor([[1,0.1,0.01],[1,1,1]]))
print(x.grad)
#######################################################
x=torch.randn(3)
x=Variable(x,requires_grad=True)
y=x*3
y.backward(torch.FloatTensor([1,1,1]))
print(x.grad)
#######################################################
#调用 backward就会自动反向求导，但是求导完之后计算图就会被丢弃，以后就不能调用backward了，为了防止这种情况，我们可以在自动求导的时候，指定不丢弃计算图retain_graph=True。
x=torch.randn(3)
x=Variable(x,requires_grad=True)
y=x*3
y.backward(torch.FloatTensor([1,1,1]),retain_graph=True)#第一次求导
print(x.grad)
x.grad.data.zero_() # 归零之前求得的梯度
y.backward(torch.FloatTensor([1,1,1]))#第二次求导
print(x.grad)

