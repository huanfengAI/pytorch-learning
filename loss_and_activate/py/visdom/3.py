from visdom import Visdom
import numpy as np
import torch
x=np.arange(0,10)
y=np.arange(0,10)*2
print(x)
viz=Visdom()
#viz.line([0.],[0.],win="first",opts=dict(title='first'))
viz.line(y,x,win="second",update='append')
