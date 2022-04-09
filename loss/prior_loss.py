import torch.nn as nn
import torch
from torchvision import models
import torchvision.transforms as tfs

import torchvision.utils as vutils
from util.utils import  RGB2HSV
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math
class prior_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1=nn.L1Loss()
        self.mse=nn.MSELoss()
    def forward(self,pred,x,y):
        x=RGB2HSV(x,flag=True)
        y=RGB2HSV(y,flag=False)
        pred=RGB2HSV(pred,flag=False)
        loss_h=self.mse(pred[:,0,:],y[:,0,:])
        delta_pred=pred[:,2,:]-pred[:,1,:]
        delta_y=y[:,2,:]-y[:,1,:]
        delta_x=x[:,2,:]-x[:,1,:]

        numerator=self.l1(delta_pred,delta_y)
        denominator=self.l1(delta_pred,delta_x)
        loss_sv=numerator/(denominator+1e-10)
        loss=loss_h+loss_sv
        return loss
