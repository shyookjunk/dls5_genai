import torch
import os
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def reverse_to_img(x):
  x=x*255
  x=x.clamp(0,255)
  x=x.to(torch.uint8)
  to_pil=transforms.ToPILImage()
  return to_pil(x)

def add_noise(x_0,t,betas):
  T=len(betas)
  assert t>=1 and t<=T
  
  alphas=1-betas
  alpha_bars=torch.cumprod(alphas,dim=0)
  t_idx=t-1
  alpha_bar=alpha_bars[t_idx]

  eps=torch.randn_like(x_0)
  x_t=torch.sqrt(alpha_bar)*x_0+torch.sqrt(1-alpha_bar)*eps
  return x_t

class Diffuser():
  def __init__(self,num_timesteps=1000,beta_start=0.0001,beta_end=0.02,device='cpu'):
    self.num_timesteps=num_timesteps
    self.device=device
    self.betas=torch.linspace(beta_start,beta_end,num_timesteps,device=device)
    self.alphas=1-self.betas
    self.alpha_bars=torch.cumprod(self.alphas,dim=0)

  def add_noise(self,x_0,t):
    T=self.num_timesteps
    assert (t>=1).all() and (t<=T).all()

    t_idx=t-1

    alpha_bar=self.alpha_bars[t_idx]
    N=alpha_bar.size(0)
    alpha_bar=alpha_bar.view(N,1,1,1)

    noise=torch.randn_like(x_0,device=self.device)
    x_t=torch.sqrt(alpha_bar)*x_0+torch.sqrt(1-alpha_bar)*noise

    return x_0,noise

  def denoise(self,model,x,t):
    T=self.num_timesteps
    assert (t>=1).all() and (t<=T).all()

    t_idx=t-1
    alpha=self.alphas[t_idx]
    alpha_bar=self.alpha_bars[t_idx]
    alpha_bar_prev=self.alpha_bars[t_idx-1]

    N=alpha.size(0)
    alpha=alpha.view(N,1,1,1)
    alpha_bar=alpha_bar.view(N,1,1,1)
    alpha_bar_prev=alpha_bar_prev.view(N,1,1,1)

    model.eval()
    with torch.no_grad():
      eps=model(x,t)
    model.train()

    noise=torch.randn_like(x,device=self.device)
    noise[t==1]=0

    mu=(x-((1-alpha)/torch.sqrt(1-alpha.bar))*eps)/torch.sqrt(alpha)
    std=torch.sqrt((1-alpha)*(1-alpha_bar_prev)/(1-alpha_bar))

    return mu+noise*std
  
  def reverse_to_img(self,x):
    x=x*255
    x=x.clamp(0,255)
    x=x.to(torch.uint8)
    x=x.cpu()
    to_pil=transforms.ToPILImage()
    return to_pil(x)
  
  def sample(self,model,x_shape=(20,1,28,28)):
    batch_size=x_shape
    x=torch.randn(x_shape,device=self.device)

    for i in tqdm(range(self.num_timesteps,0,-1)):
      t=torch.tensor([i]*batch_size,device=self.device,dtype=torch.long)
      x=self.denoise(model,x,t)

    images=[self.reverse_to_img(x[i]} for i in range(batch_size)]
    return images

# read image
current_dir=os.path.dirname(os.path.abspath(__file__))
file_path=os.path.join(current_dir,'flower.png')
image=plt.imread(file_path)
print(image.shape)

# preprocess the image
preprocess=transforms.ToTensor()
x=preprocess(image)
print(x.shape)

T=1000
beta_start=0.0001
beta_end=0.02
betas=torch.linspace(beta_start,beta_end,T)

t=100
x_t=add_noise(x,t,betas)

img=reverse_to_img(x_t)
plt.imshow(img)
plt.title(f'Noise: {t}')
plt.axis('off')
plt.show()

#imgs=[]

# for t in range(T):
#   if t%100==0:
#     img=reverse_to_img(x)
#     imgs.append(img)

#   beta=betas[t]
#   eps=torch.randn_like(x)
#   x=torch.sqrt(1-beta)*x+torch.sqrt(beta)*eps

# plt.figure(figsize=(15,6))
# for i,img in enumerate(imgs[:10]):
#   plt.subplot(2,5,i+1)
#   plt.imshow(img)
#   plt.title(f'Noise: {i*100}')
#   plt.axis('off')

# plt.show()


# x=torch.randn(3,64,64)
# T=1000
# betas=torch.linspace(0.0001,0.02,T)

# for t in range(T):
#   beta=betas[t]
#   eps=torch.randn_like(x)
#   x=torch.sqrt(1-beta)*x+torch.sqrt(beta)*eps
