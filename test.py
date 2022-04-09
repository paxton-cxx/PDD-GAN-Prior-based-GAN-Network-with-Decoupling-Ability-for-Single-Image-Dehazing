import os
import argparse
from dataset.data_test_sots_2 import RESIDE_Dataset
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import torchvision.utils as vutils
from tqdm import tqdm
from main import my_GAN
from metric import ssim,psnr
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from datetime import datetime,timedelta
import pyiqa
parser = argparse.ArgumentParser(description='image-dehazing')
torch.autograd.set_detect_anomaly(True)
parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
ITS_dataset = RESIDE_Dataset(size=224,choice=4)
ITS_dataloader=torch.utils.data.DataLoader(ITS_dataset,batch_size=args.batch_size,shuffle=False,num_workers=10,drop_last=False)
model = my_GAN.load_from_checkpoint('./checkpoint/best.ckpt').to('cuda')
psnr_metric = pyiqa.create_metric('psnr')
ssim_metric = pyiqa.create_metric('ssim')
vif_metric = pyiqa.create_metric('vif')
fsim_metric = pyiqa.create_metric('fsim')
def test():
    psnr_all=[]
    ssim_all=[]
    vif_all=[]
    fsim_all=[]
    
    for x,y,_ in ITS_dataloader:
        x=x.to('cuda')
        y=y.to('cuda')
        with torch.no_grad():
            pred=model(x)    
            pred=pred[:,:3,:,:]
            y=y[:,:3,:,:]

            # ssim1=ssim(pred,y).item()
            # psnr1=psnr(pred,y).item()
            psnr_step=psnr_metric(pred,y).mean().item()
            ssim_step=ssim_metric(pred,y).mean().item()
            vif_step=vif_metric(pred,y).mean().item()
            fsim_step=fsim_metric(pred,y).mean().item()
            print('ssim:{};psnr:{};vif:{};fsim:{}'.format(ssim_step,psnr_step,vif_step,fsim_step))
            ssim_all.append(ssim_step)
            psnr_all.append(psnr_step)
            vif_all.append(vif_step)
            fsim_all.append(fsim_step)
            # print("ssim:{}".format(ssim1))
            # print("psnr:{}".format(psnr1))
    print('*****************************')
    ssim_mean=np.array(ssim_all).mean()
    psnr_mean=np.array(psnr_all).mean()
    vif_mean=np.array(vif_all).mean()
    fsim_mean=np.array(fsim_all).mean()
    print('ssim:{};psnr:{};vif:{};fsim:{}'.format(ssim_mean,psnr_mean,vif_mean,fsim_mean))

if __name__ == '__main__':
    seed=43
    pl.seed_everything(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    test()

    
    
    
