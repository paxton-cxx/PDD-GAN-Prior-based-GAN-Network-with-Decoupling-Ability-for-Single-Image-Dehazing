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
    for x,y,name in ITS_dataloader:
        x=x.to('cuda')
        y=y.to('cuda')
        with torch.no_grad():
            pred=model(x)    
            pred=pred[:,:3,:,:]
            vutils.save_image(pred, output_dir + '/' + name + '_pred.png')

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
    output_dir='./test_img_result'
    test()


    
    
    
