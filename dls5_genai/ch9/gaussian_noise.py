import torch
import os
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# read image
current_dir=os.path.dirname(os.path.abspath(__file__))
file_path=os.path.join(current_dir,'flower.png')
image=plt.imread(file_path)
print(image.shape)

# preprocess the image
preprocess=transforms.ToTensor()
x=preprocess(image)
print(x.shape)

# x=torch.randn(3,64,64)
# T=1000
# betas=torch.linspace(0.0001,0.02,T)

# for t in range(T):
#   beta=betas[t]
#   eps=torch.randn_like(x)
#   x=torch.sqrt(1-beta)*x+torch.sqrt(beta)*eps
