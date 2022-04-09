import torch.nn as nn
from torchvision import models
import torch
import torchvision.transforms as tfs
import random
import torch.nn.functional as F
from dataset.data_final import RESIDE_Dataset

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1) 
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4) 
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

class perceptual_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = Vgg19()
        self.loss_mse=nn.MSELoss()
        self.weights = [1.0, 1.0, 1.0, 1.0,1.0]

    def forward(self, pred,ground_truth):
        pred=pred[:,:3,:]
        ground_truth=ground_truth[:,:3,:]
        y_c_features = self.vgg(pred)
        y_hat_features = self.vgg(ground_truth)

        content_loss=0
        tv_loss=0
        style_loss = 0.0
        # calculate style loss
        y_gram=[gram(fmap) for fmap in y_c_features]
        y_hat_gram = [gram(fmap) for fmap in  y_hat_features]
 
        for j in range(5):
            style_loss1 = self.loss_mse(y_gram[j], y_hat_gram[j][:len(pred)])
            style_loss += style_loss1

        # calculate content loss
        for i in range(5):
            recon = y_c_features[i]      
            recon_hat = y_hat_features[i]
            content_loss1 = self.weights[i]*self.loss_mse(recon_hat, recon)
            content_loss += content_loss1

        # calculate total variation regularization (anisotropic version)
        
        # diff_i = torch.sum(torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1]))
        # diff_j = torch.sum(torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :]))
        # tv_loss = (diff_i + diff_j)
        # tv_loss += tv_loss

        # total loss
        # total_loss = (1e5*style_loss + content_loss)
        total_loss = style_loss + content_loss 
    

        return total_loss
def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w*h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G


if __name__=="__main__":
    ITS_dataset = RESIDE_Dataset(size=224,choice=2)
    ITS_dataloader=torch.utils.data.DataLoader(ITS_dataset,batch_size=3,shuffle=True,num_workers=10,drop_last=True)

    for i,(x,y) in enumerate(ITS_dataloader):
        pred=perceptual_loss()(x,y)
        print(pred)
        pred=perceptual_loss()(y,y)
        print(pred)
        # print(len(x))
        if i==3:
            break