import os
import os.path
import torch
import numpy as np
import cv2
import torchvision.transforms as tfs
import torchvision.utils as vutils
from torchvision import transforms
import torch.nn.functional as F
import math
def tensor_to_rgb(x):
    output = x.cpu()
    output = output.data.squeeze(0).numpy()
    output = (output + 1.0) * 127.5
    output = output.clip(0, 255).transpose(1, 2, 0)
    return output


def rgb_to_tensor(x):
    output = (transforms.ToTensor()(x) - 0.5) / 0.5
    return output


def RGB2HSV(pred,flag=True):
    
    for i in range(len(pred)):
        img=pred[i,:3,:]
        if flag:
            img=tensor2im(img)
        else:
            img=tfs.ToPILImage()(img)
            img=np.array(img)
        img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        img=tfs.ToTensor()(img)
        img=img[None,:]
        if i==0:
            img_hsv=img
        else:
            img_hsv=torch.cat([img_hsv,img],dim=0)
    img_hsv=img_hsv.to(pred.device)
    return img_hsv


class ImagePool:
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        if pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, image):
        if self.pool_size == 0:
            return image
        if self.num_imgs < self.pool_size:
            self.images.append(image.clone())
            self.num_imgs += 1
            return image
        else:
            if np.random.uniform(0, 1) > 0.5:
                random_id = np.random.randint(self.pool_size, size=1)[0]
                tmp = self.images[random_id].clone()
                self.images[random_id] = image.clone()
                return tmp
            else:
                return image


def tensor2im(input_image, imtype=np.uint16):

    """"transform tensor to numpy img

    Parameters:
        input_image (tensor) --  tensor img
        imtype (type)        --  numpy
    # """

    mean=[0.58072233, 0.60468274, 0.6249737]
    std=[0.15053451, 0.15662293, 0.17608616]
    image_numpy = input_image  # convert it into a numpy array

    for i in range(len(mean)):
        image_numpy[i] = image_numpy[i] * std[i] + mean[i]
    image_numpy=tfs.ToPILImage()(image_numpy)
    image_numpy=np.array(image_numpy)
    return image_numpy

