import torchvision.models as models
#加载torchvision中已经存在的模型，需要下载 
#resnet18 = models.resnet18(pretrained=True) 
print(1)
model = models.vgg16(pretrained=True)
print(2) 
#alexnet = models.alexnet(pretrained=True)
features = t.Tensor()
def hook(module, input, output):
    features.copy_(output.data)
input=torch.arange(0,12).view(3,4)   
handle = model.layer8.register_forward_hook(hook)
_ = model(input)
# # 用完hook后删除
handle.remove()
