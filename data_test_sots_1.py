import os
import random
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
import math
#ITS_DATA
path_ITS_small="indoor_path"
path_OTS_small="outdoor_path"


ITS_haze_path=[]
OTS_haze_path=[]
ITS_clear_path=[]
OTS_clear_path=[]

def bandstop_filter(image, radius=25, w=60, n=1):
    """
    band-stop filter
    :param image: the input
    :param radius: the distance between center point to origin
    :param w: bandwidth
    :param n: number of order
    :return: results
    """
    fft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dshift = np.fft.fftshift(fft)
    rows, cols = image.shape[:2]
    mid_row, mid_col = int(rows // 2), int(cols // 2)
    mask = np.zeros((rows, cols, 2), np.float32)
    for i in range(0, rows):
        for j in range(0, cols):

            d = math.sqrt(pow(i - mid_row, 2) + pow(j - mid_col, 2))
            if radius - w / 2 < d < radius + w / 2:
                mask[i, j, 0] = mask[i, j, 1] = 0
            else:
                mask[i, j, 0] = mask[i, j, 1] = 1

    fft_filtering = dshift * np.float32(mask)
    ishift = np.fft.ifftshift(fft_filtering)
    image_filtering = cv2.idft(ishift)
    image_filtering = cv2.magnitude(image_filtering[:, :, 0], image_filtering[:, :, 1])
    cv2.normalize(image_filtering, image_filtering, 0, 1, cv2.NORM_MINMAX)
    img_new = torch.from_numpy(image_filtering)

    return img_new
def frequency_trans(img):
    img_man=img
    rows,cols = img_man.shape
    mask1 = np.ones(img_man.shape,np.uint8)
    mask1[rows//2-8:rows//2+8,cols//2-8:cols//2+8] = 0
    mask2 = np.zeros(img_man.shape,np.uint8)
    mask2[rows//2-80:rows//2+80,cols//2-80:cols//2+80] = 1
    mask = mask1*mask2
    #--------------------------------
    f1 = np.fft.fft2(img_man)
    f1shift = np.fft.fftshift(f1)
    f1shift = f1shift*mask
    f2shift = np.fft.ifftshift(f1shift)
    img_new = np.fft.ifft2(f2shift)
    img_new = np.abs(img_new)
    img_new = (img_new-np.amin(img_new))/(np.amax(img_new)-np.amin(img_new))
    img_new=torch.from_numpy(img_new)
    return img_new

def RESIDE_path_small(path_ITS=path_ITS_small,path_OTS=path_OTS_small):
    OTS_haze=[]
    ITS_haze=[]
    OTS_clear=[]
    ITS_clear=[]
    path_ITS_haze=os.path.join(path_ITS,"haze")
    path_ITS_clear=os.path.join(path_ITS,"clear")
    path_OTS_haze=os.path.join(path_OTS,"haze")
    path_OTS_clear=os.path.join(path_OTS,"clear")

    for root,dirs,files in os.walk(path_ITS_haze):
        for file in files:
            ITS_haze.append(os.path.join(root,file))

    for root,dirs,files in os.walk(path_ITS_clear):
        for file in files:
            ITS_clear.append(os.path.join(root,file))

    for root,dirs,files in os.walk(path_OTS_haze):
        for file in files:
            OTS_haze.append(os.path.join(root,file))

    for root,dirs,files in os.walk(path_OTS_clear):
        for file in files:
            OTS_clear.append(os.path.join(root,file))    
    return ITS_haze,ITS_clear,OTS_haze,OTS_clear
ITS_haze_path_small,ITS_clear_path_small,OTS_haze_path_small,OTS_clear_path_small=RESIDE_path_small()

class RESIDE_Dataset(data.Dataset):
    def __init__(self,choice=4,size=224):#indoor=3,outdoor=4
        super(RESIDE_Dataset,self).__init__()
        self.size=size
        self.choice=choice
        
        if choice==3:
            self.haze=ITS_haze_path_small
            self.clear=ITS_clear_path_small
            self.mean=[0.58072233, 0.60468274, 0.6249737]
            self.std=[0.15053451, 0.15662293, 0.17608616]

        elif choice==4:  
            self.haze=OTS_haze_path_small
            self.clear=OTS_clear_path_small
            self.mean=[0.58072233, 0.60468274, 0.6249737]
            self.std=[0.15053451, 0.15662293, 0.17608616]

    def __getitem__(self, index) :
        pattern=re.compile(r'.*haze')  
        haze=Image.open(self.haze[index])
        haze_id=self.haze[index].split('/')[-1].split('_')[0]
        id=haze_id
        last=self.haze[index].split('/')[-1].split('.')[-1]
        haze_id=haze_id+'.'+last
        clear_path=pattern.match(self.haze[index]).group()
        clear_path=clear_path.replace('haze','clear')
        if self.choice==4:
            haze_id=haze_id.replace('jpg','png')
        clear_path=os.path.join(clear_path,haze_id)
        
        clear=Image.open(clear_path)
        clear1=np.array(clear)
       
        clear_4=cv2.imread(clear_path,0)
        haze_4=cv2.imread(self.haze[index],0)

        haze,clear=self.augData(haze ,clear,haze_4,clear_4)
        return haze,clear,id
        
    def augData(self,data,target,haze_4,clear_4):

        trans_data=tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean=self.mean, std= self.std)
        ])

        trans_target=tfs.Compose([tfs.ToTensor()])
        data1=trans_data(data)
        target1=trans_target(target)
        data_hl=bandstop_filter(haze_4)[None,:]
        data=torch.cat([data1,data_hl],0)

        target1=target1[:3,:]
        target_hl=bandstop_filter(clear_4)[None,:]
        target=torch.cat([target1,target_hl],0)

        a1=data.shape[-1]
        a2=data.shape[-2]
        b1=target.shape[-1]
        b2=target.shape[-2]
        
        if a1%16!=0:
            a1=(a1//16+1)*16
            data=tfs.Resize((a2,a1))(data)
        if a2%16!=0:
            a2=(a2+1)//16*16
            data=tfs.Resize((a2,a1))(data)

        if b1%16!=0:
            b1=(b1//16+1)*16
            target=tfs.Resize((b2,b1))(target)
        if b2%16!=0:
            b2=(b2//16+1)*16
            target=tfs.Resize((b2,b1))(target)
        return  data.float(),target.float()
    
    def __len__(self):
        return len(self.haze)

if __name__ == '__main__':
    ITS_dataset = RESIDE_Dataset(choice=4)
  
    print(len(ITS_dataset))
    ITS_dataloader=torch.utils.data.DataLoader(ITS_dataset,batch_size=1,shuffle=True,num_workers=10)
    for x,y,_ in ITS_dataloader:
        # x=x
        print(x.shape,y.shape)
        break
    # print(OTS_haze_path_small[:4])



