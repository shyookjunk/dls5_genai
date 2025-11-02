import torch
from torch import nn

class ConvBlock(nn.Module):
  def __init__(self,in_ch,out_ch):
    super().__init__()
    self.convs=nn.Sequential(
                nn.Conv2d(in_ch,out_ch,3,padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(out_ch,out_ch,3,padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU())
  def forward(self,x):
    return self.convs(x)
  
class UNet(nn.Module):
  def __init__(self,in_ch=1):
    super().__init__()
    self.down1=ConvBlock(in_ch,64)
    self.down2=ConvBlock(64,128)
    self.bot2=ConvBlock(128,256)
    self.up2=ConvBlock(128+256,128)
    self.up1=ConvBlock(128+64,64)
    self.out=nn.Conv2d(64,in_ch,1)

    self.maxpool=nn.MaxPool2d(2)
    self.upsample=nn.Upsample(scale_factor=2,mode='bilinear')

  def forward(self,x):
    x1=self.down1(x)
    x-self.maxplool(x1)
    x2=self.down2(x)
    x=self.maxplool(x2)
    
    x=self.bot1(x)