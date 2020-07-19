#可视化工具tensorboard和visdom
#Tensorboard也是一个相对独立的工具，只要用户保存的数据遵循相应的格式，tensorboard就能读取这些数据并进行可视化。
#TensorboardX是将Tensorboard的功能抽取出来，使得非TensorFlow用户也能使用它进行可视化，几乎支持原生TensorBoard的全部功能。
#要想使用tensorboard需要安装tensorflow
#然后安装安装tensorboard: `pip install tensorboard`
# - 安装tensorboardX：可通过`pip install tensorboardX`命令直接安装。

##使用tensorboard --logdir experimient_cnn --port 6006
#其中experimient_cnn为数据保存的数据遵循的相应的格式
#port为端口号http://localhost:6006看到效果了
#这样我们就可以使用
from tensorboardX import SummaryWriter
# 构建logger对象，logdir用来指定log文件的保存路径
# flush_secs用来指定刷新同步间隔
logger=SummaryWriter(log_dir='experimient_cnn',flush_secs=2)
for ii in range(100):
	logger.add_scalar('data/loss',10-ii**0.5)
	logger.add_scalar('data/accuracy',ii**0.5/10)
 #左侧的Horizontal Axis下有三个选项，分别是：
# - Step：根据步长来记录，log_value时如果有步长，则将其作为x轴坐标描点画线。
# - Relative：用前后相对顺序描点画线，可认为logger自己维护了一个`step`属性，每调用一次log_value就自动加１。
# - Wall：按时间排序描点画线。
# 
# 左侧的Smoothing条可以左右拖动，用来调节平滑的幅度。点击右上角的刷新按钮可立即刷新结果，默认是每30s自动刷新数据。可见tensorboard_logger的使用十分简单，但它只能统计简单的数值信息，不支持其它功能。


#visdom
#安装pip install visdom
#安装完成后，需通过`python -m visdom.server`命令启动visdom服务，或通过`nohup python -m visdom.server &`命令将服务放至后台运行。Visdom服务是一个web server服务，默认绑定8097端口，客户端与服务器间通过tornado进行非阻塞交互。
# Visdom中有两个重要概念：
# - env：环境。不同环境的可视化结果相互隔离，互不影响，在使用时如果不指定env，默认使用`main`。不同用户、不同程序一般使用不同的env。
# - pane：窗格。窗格可用于可视化图像、数值或打印文本等，其可以拖动、缩放、保存和关闭。一个程序中可使用同一个env中的不同pane，每个pane可视化或记录某一信息。
# 
# 如图4所示，当前env共有两个pane，一个用于打印log，另一个用于记录损失函数的变化。点击clear按钮可以清空当前env的所有pane，点击save按钮可将当前env保存成json文件，保存路径位于`~/.visdom/`目录下。也可修改env的名字后点击fork，保存当前env的状态至更名后的env。

# Visdom的使用有两点需要注意的地方：
# - 需手动指定保存env，可在web界面点击save按钮或在程序中调用save方法，否则visdom服务重启后，env等信息会丢失。
# - 客户端与服务器之间的交互采用tornado异步框架，可视化操作不会阻塞当前程序，网络异常也不会导致程序退出。
# 
# Visdom以Plotly为基础，支持丰富的可视化操作，下面举例说明一些最常用的操作。

import visdom
#新建一个连接客户端
# 指定env = u'test1'，默认端口为8097，host是‘localhost'

vis=visdom.Visdom(env=u'test1' ,use_incoming_socket=False)
x=torch.arange(1,30,0.01)
y=torch.sin(x)
vis.line(X=x,Y=y,win='sinx',opts={'title': 'y=sin(x)'})
# - vis = visdom.Visdom(env=u'test1')，用于构建一个客户端，客户端除指定env之外，还可以指定host、port等参数。
# - vis作为一个客户端对象，可以使用常见的画图函数，包括：
# 
#     - line：类似Matlab中的`plot`操作，用于记录某些标量的变化，如损失、准确率等
#     - image：可视化图片，可以是输入的图片，也可以是GAN生成的图片，还可以是卷积核的信息
#     - text：用于记录日志等文字信息，支持html格式
#     - histgram：可视化分布，主要是查看数据、参数的分布
#     - scatter：绘制散点图
#     - bar：绘制柱状图
#     - pie：绘制饼状图
#     - 更多操作可参考visdom的github主页
#     
# 这里主要介绍深度学习中常见的line、image和text操作。
# 
# Visdom同时支持PyTorch的tensor和Numpy的ndarray两种数据结构，但不支持Python的int、float等类型，因此每次传入时都需先将数据转成ndarray或tensor。上述操作的参数一般不同，但有两个参数是绝大多数操作都具备的：
# - win：用于指定pane的名字，如果不指定，visdom将自动分配一个新的pane。如果两次操作指定的win名字一样，新的操作将覆盖当前pane的内容，因此建议每次操作都重新指定win。
# - opts：选项，接收一个字典，常见的option包括`title`、`xlabel`、`ylabel`、`width`等，主要用于设置pane的显示格式。
# 
# 之前提到过，每次操作都会覆盖之前的数值，但往往我们在训练网络的过程中需不断更新数值，如损失值等，这时就需要指定参数`update='append'`来避免覆盖之前的数值。而除了使用update参数以外，还可以使用`vis.updateTrace`方法来更新图，但`updateTrace`不仅能在指定pane上新增一个和已有数据相互独立的Trace，还能像`update='append'`那样在同一条trace上追加数据。
for ii in range(0, 10):
    # y = x
    x = torch.Tensor([ii])
    y = x
    vis.line(X=x, Y=y, win='polynomial', update='append' if ii>0 else None)
    
# updateTrace 新增一条线
x = torch.arange(0, 9, 0.1)
y = (x ** 2) / 9
vis.line(X=x, Y=y, win='polynomial', name='this is a new Trace',update='new')

## 打开浏览器，输入`http://localhost:8097`，可以看到如图6所示的结果。
# ![图6 ：append和updateTrace可视化效果 ](imgs/visdom_update.svg)

# image的画图功能可分为如下两类：
# - `image`接收一个二维或三维向量，$H\times W$或$3 \times H\times W$，前者是黑白图像，后者是彩色图像。
# - `images`接收一个四维向量$N\times C\times H\times W$，$C$可以是1或3，分别代表黑白和彩色图像。可实现类似torchvision中make_grid的功能，将多张图片拼接在一起。`images`也可以接收一个二维或三维的向量，此时它所实现的功能与image一致。

# In[37]:


# 可视化一个随机的黑白图片
#vis.image(t.randn(64, 64).numpy())

# 随机可视化一张彩色图片
#vis.image(t.randn(3, 64, 64).numpy(), win='random2')

# 可视化36张随机的彩色图片，每一行6张
#vis.images(t.randn(36, 3, 64, 64).numpy(), nrow=6, win='random3', opts={'title':'random_imgs'})


# 其中images的可视化输出如图7所示。
# ![图7： images可视化输出](imgs/visdom_images.png)

# `vis.text`用于可视化文本，支持所有的html标签，同时也遵循着html的语法标准。例如，换行需使用`<br>`标签，`\r\n`无法实现换行。下面举例说明。

# In[38]:


#vis.text(u'''<h1>Hello Visdom</h1><br>Visdom是Facebook专门为<b>PyTorch</b>开发的一个可视化工具，
#         在内部使用了很久，在2017年3月份开源了它。
#         
 #        Visdom十分轻量级，但是却有十分强大的功能，支持几乎所有的科学运算可视化任务''',
 #        win='visdom',
 #        opts={'title': u'visdom简介' }
#        )


# ![图8：text的可视化输出](imgs/visdom_text.png)
