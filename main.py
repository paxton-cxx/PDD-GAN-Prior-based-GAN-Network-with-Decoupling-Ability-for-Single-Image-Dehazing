import os
import argparse
from model.PDD import Generator, Discriminator
from dataset.data_final import RESIDE_Dataset
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
from util.metric import ssim,psnr
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from datetime import datetime,timedelta
from loss.perceptual_loss import perceptual_loss
from loss.prior_loss import prior_loss
parser = argparse.ArgumentParser(description='image-dehazing')
torch.autograd.set_detect_anomaly(True)
parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')
# optimization
parser.add_argument('--p_factor', type=float, default=1, help='perceptual loss factor')
parser.add_argument('--c_factor', type=float, default=0.5, help='contrast loss factor')
parser.add_argument('--hsv_factor', type=float, default=1, help='hsv loss factor')
parser.add_argument('--g_factor', type=float, default=0.5, help='gan loss factor')

parser.add_argument('--glr', type=float, default=0.0001, help='generator learning rate')
parser.add_argument('--dlr', type=float, default=0.0001, help='discriminator learning rate')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')

parser.add_argument('--patch_gan', type=int, default=28, help='Path GAN size')
parser.add_argument('--pool_size', type=int, default=28, help='Buffer size for storing generated samples from G')

args = parser.parse_args()


class my_GAN(pl.LightningModule):
    def __init__(self,lrg=args.glr,lrd=args.dlr,max_epochs=100,weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.netG=Generator()
        self.netD = Discriminator()
        
        self.l1_loss = nn.L1Loss()
        self.l2_loss = perceptual_loss()
        self.bce_loss = nn.BCELoss()
        self.prior_loss=prior_loss()
        # networks

    def forward(self, x):
        return self.netG(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        input_image, target_image,_= batch
        output_image = self.netG(input_image)
        
        # train generator
        if optimizer_idx == 0:

            self.real_label = torch.ones([input_image.shape[0], 1, input_image.shape[2]//8, input_image.shape[3]//8]).to('cuda')
            self.fake_label = torch.zeros([input_image.shape[0], 1, input_image.shape[2]//8,input_image.shape[3]//8]).to('cuda')
            ## reconstruction loss
            g_res_loss = self.l1_loss(output_image, target_image)
    

            ## perceptual loss
            g_per_loss = args.p_factor * self.l2_loss(output_image[:,:3,:], target_image[:,:3,:])
    

            ## gan loss
            output = self.netD(output_image)
            g_gan_loss = args.g_factor * self.bce_loss(output, self.real_label)

            ## prior loss
            
            g_prior_loss=args.hsv_factor*self.prior_loss(output_image,input_image.clone(),target_image)


            ## loss
            g_total_loss = g_res_loss + g_gan_loss + g_per_loss+g_prior_loss
            pred=output_image
            
            ssim1=ssim(pred[:,:3,:],target_image[:,:3,:]).item()
            psnr1=psnr(pred[:,:3,:],target_image[:,:3,:])
            self.log('ssim',ssim1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('psnr',psnr1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("g_total_loss", g_total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            # self.log("d_total_loss", d_total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            return  g_total_loss
        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples
            self.real_label = torch.ones([input_image.shape[0], 1, input_image.shape[2]//8, input_image.shape[3]//8]).to('cuda')
            self.fake_label = torch.zeros([input_image.shape[0], 1, input_image.shape[2]//8,input_image.shape[3]//8]).to('cuda')
            # how well can it label as real?
            real_output = self.netD(target_image)
            d_real_loss = self.bce_loss(real_output, self.real_label)
          

            ## fake image
            fake_image = output_image.detach()
            # fake_image = Variable(self.image_pool.query(fake_image.data))
            fake_output = self.netD(fake_image)
            d_fake_loss = self.bce_loss(fake_output, self.fake_label)
  

            ## loss
            d_total_loss = d_real_loss + d_fake_loss

            
            pred=self(input_image)
            ssim1=ssim(pred[:,:3,:],target_image[:,:3,:]).item()
            psnr1=psnr(pred[:,:3,:],target_image[:,:3,:])
            self.log('ssim',ssim1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('psnr',psnr1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            # self.log("g_total_loss", g_total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("d_total_loss", d_total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return d_total_loss

         
            

    def configure_optimizers(self):
        opt_g = optim.AdamW(self.parameters(), lr=self.hparams.lrg, weight_decay=self.hparams.weight_decay)
        # lr_g = optim.lr_scheduler.CosineAnnealingLR(
        #     opt_g, T_max=self.hparams.max_epochs, eta_min=self.hparams.lrg/ 50
        # )

        opt_d = optim.AdamW(self.parameters(), lr=self.hparams.lrd, weight_decay=self.hparams.weight_decay)
        # lr_d = optim.lr_scheduler.CosineAnnealingLR(
        #     opt_d, T_max=self.hparams.max_epochs, eta_min=self.hparams.lrd/ 50
        # )
        return [opt_g, opt_d], []



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

    logger = TensorBoardLogger('tb_logs', name='PDD_GAN')
    ITS_dataset = RESIDE_Dataset(size=224,choice=4)
    ITS_dataloader=torch.utils.data.DataLoader(ITS_dataset,batch_size=args.batch_size,shuffle=True,num_workers=10,drop_last=True)
    timespan=timedelta(seconds=30)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    
    model = my_GAN()
    trainer = pl.Trainer(default_root_dir=os.path.join('save_location', "checkpoint"),
                         accelerator ='ddp',gpus=4,logger=logger,max_epochs=100,
                        callbacks=[
            ModelCheckpoint(save_weights_only=True,train_time_interval=timespan,save_top_k=-1),
            LearningRateMonitor("epoch")],
            progress_bar_refresh_rate=1)

    trainer.fit(model, ITS_dataloader)
    
    
