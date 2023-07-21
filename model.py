from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class CFAR_Resnet_Custom(nn.Module):
  def __init__(self):
    super(CFAR_Resnet_Custom, self).__init__()

        # Prep Layer
    self.prep_layer = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1,stride=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU())
    # ************************************* Layer 1 ********************************
    # Layer 1
    self.lyr1 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1,stride=1, bias=False),
        nn.MaxPool2d(2, 2),
        nn.BatchNorm2d(128),
        nn.ReLU())

    # Res Block 1
    self.res1= nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1,stride=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(),

        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1,stride=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(),

        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1,stride=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU())

    # ************************************** Layer 2 *********************************
    # Layer 2
    self.lyr2 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1,stride=1, bias=False),
        nn.MaxPool2d(2, 2),
        nn.BatchNorm2d(256),
        nn.ReLU())

    # ************************************** Layer 3 **********************************
    # Layer 3

    self.lyr3 = nn.Sequential(
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1,stride=1, bias=False),
        nn.MaxPool2d(2, 2),
        nn.BatchNorm2d(512),
        nn.ReLU())

    self.res2 = nn.Sequential(
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1,stride=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(),

        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1,stride=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(),

        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1,stride=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU())
    # ************************************ Max Pool Layer *******************************

    self.maxpool = nn.MaxPool2d(kernel_size=4, stride=2)


    # ************************************ FC Layer *************************************
    self.fc=nn.Sequential(
      nn.Linear(1*1*512,40),
      nn.Linear(40,10))



  def forward(self, x):
      x = self.prep_layer(x) # O/P SIZE - 32
      x = self.lyr1(x) # O/P SIZE - 16
      R1= self.res1(x) # O/P SIZE - 16
      x = x+ R1 # O/P SIZE - 16
      x = self.lyr2(x) # O/P SIZE - 8
      x = self.lyr3(x) # O/P Sixe - 4
      R2= self.res2(x) # O/P Size - 4
      x = x + R2 # O/P Size - 8
      x=  self.maxpool(x) # O/P Size -3
      x = x.view(-1, 1*1*512)
      x = self.fc(x)
      x = x.view(-1, 10)
      return F.log_softmax(x, dim=-1)
