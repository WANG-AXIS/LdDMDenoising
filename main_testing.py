#!/usr/bin/env python
# coding: utf-8



import torch
import os
import time
import h5py
import numpy as np
from timeit import default_timer as timer
from sklearn.utils import shuffle
import logging
import torchvision
from torchvision import models
from torch import nn
from torch import autograd
from torch import optim
from torch.autograd import grad
import  torch.nn.functional as F
import pytorch_ssim
import torch.nn.init as init
from torch.autograd import Variable
import pydicom as dicom
import matplotlib.pyplot as plt


#get_ipython().run_line_magic('matplotlib', 'inline')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

##################Custom setting###############


Model_folder = 'final_models/'

lambda_mse = 20
lambda_ssim = 0.25
lambda_adv = 0


cases = ['L_CC', 'L_MLO', 'R_CC', 'R_MLO']



###############################################


    
###############################################



class BasicBlock(nn.Module):

    def __init__(self, num_filters = 64):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out



class Generator(nn.Module):
    def __init__(self, num_filters=64):
        super(Generator, self).__init__()
        
        self.conv_first = nn.Conv2d(1, num_filters, 3, 1, 1)
        
        self.block1 = BasicBlock(num_filters)
        self.block2 = BasicBlock(num_filters)
        self.block3 = BasicBlock(num_filters)
        self.block4 = BasicBlock(num_filters)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.conv_last = nn.Conv2d(num_filters, 1, 3, 1, 1)
        
    def forward(self, x):
        
        identity = x
        out1 = self.conv_first(x)
        
        out2 = self.block1(out1)
        
        out3 = self.block2(out2)
        
        out3 = self.relu(out3 + out1)
        
        out4 = self.block3(out3)
        
        out5 = self.block4(out4)
        
        out5 = self.relu(out5 + out3)
        
        out = self.conv_last(out5)
        
        out = self.relu(identity + out)
        
        return out
    

    
    
netG = Generator()


netG = netG.to(device)

netG.eval()

def scale(x, vmin=0, vmax=8000.0, factor=1.):
    # [-1,1]
    x = x / vmax / factor
    x[x > 1.0] = 1.0
    x[x < 0.0] = 0.0
    #x = 2. * x - 1.0
    x = np.expand_dims(x, axis=1)
    return x.astype(np.float32)

def de_scale(x):
    # [-1,1]
    
    x = x * 8000
    
    x = np.squeeze(x)
    return x.astype(np.int16)



    
def crop_image(img, scale_factor):
    # for normal dose
    mask = img / scale_factor < 15000
    
    # w
    mask_w = np.sum(mask, 1) > 0
    res = np.where(mask_w == True)
    w_min, w_max = res[0][0], res[0][-1]
    
    # h
    mask_h = np.sum(mask, 0) > 0
    res = np.where(mask_h == True)
    h_min, h_max = res[0][0], res[0][-1]
    
    # padding
    if h_min == 0:
        h_max += 32
    if h_max == 0:
        h_min -= 32
        
    if (h_max - h_min) %2 == 1:
        h_max += 1
    if (w_max - w_min) %2 == 1:
        w_max += 1
    
    return w_min, h_min, w_max, h_max



ssim_ssim_loss = pytorch_ssim.SSIM(window_size = 11)



def size_map(cropped_image):
    print('map:cropped_size', cropped_image.shape)
    w, h = cropped_image.shape
    w_pad = (w // 64 + 1) * 64
    h_pad = (h // 64 + 1) * 64
    
    print((w_pad - w)//2, (h_pad - h)//2)
    
    padding = (((w_pad - w)//2 + 64, (w_pad - w)//2 + 64), 
               ((h_pad - h)//2 + 64, (h_pad - h)//2 + 64))
    
    padded_image = np.pad(cropped_image, padding, mode='reflect')
    
    print('map:padded_size', padded_image.shape)
    
    res_list = []
    
    n_w = w_pad // 64 
    n_h = h_pad // 64 
    
    
    
    for i in range(n_w):
        for j in range(n_h):
            patch = padded_image[i*64: (i+3)*64, j*64:(j+3)*64]
            res_list.append((i,j,patch))
    return res_list, (w,h), padded_image.shape
    
            
            
               


def size_reduce(res_list, original_size, padded_size):
    print('reduce:padded_size', padded_size)
    print('reduce:origin_size', original_size)
    
    final_image = np.ones((padded_size))
    for res in res_list:
        final_image[(res[0]+1)*64:(res[0]+2)*64, (res[1]+1)*64:(res[1]+2)*64] = res[2][64:128,64:128]
    org_w, org_h = original_size
    pad_w, pad_h = padded_size
    
    start_w = (pad_w - org_w) // 2
    start_h = (pad_h - org_h) // 2
    
    print(start_w, start_h)
    
    return final_image[start_w:start_w+org_w, start_h:start_h+org_h]

        


def test(batch_data, noise_level, scale_factor):

        start_time = time.time()
        start = timer()
        
        w, h = batch_data.shape
        
        w_min, h_min, w_max, h_max = crop_image(batch_data)
        batch_data = batch_data[w_min:w_max, h_min:h_max]
        
        batch_data = np.expand_dims(batch_data, 0)
        
        batch_data = torch.from_numpy(scale(batch_data, factor=scale_factor)).to(device)
        appendix = 'perceptual4_{}'.format(noise_level)
        netG.load_state_dict(torch.load(Model_folder + 'model_ResDial_{}.pth'.format(appendix)))

        with torch.no_grad():
            fake_label = netG(batch_data)

        end = timer()
        
        print('Processing time is: %.6f' %(end-start))
        
        fake_label = fake_label.to('cpu').numpy()
        
        res = np.ones((w, h)) 
        res[w_min:w_max, h_min:h_max] = fake_label
        
        return res
        
        
def test_vary_size(batch_data, noise_level, scale_factor, loss_name):

        start_time = time.time()
        start = timer()
        
        w, h = batch_data.shape
        
        res_list, org_size, pad_size = size_map(batch_data)
        
        new_res_list = []
        
        appendix = '{}_{}'.format(loss_name, noise_level)
        
        state_dict = torch.load(Model_folder + 'model_ResResNet_{}.pth'.format(appendix), map_location=device)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module' in k:
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        # load params
        netG.load_state_dict(new_state_dict)
        
        
        
        for res in res_list:
            
        
            batch_data = np.expand_dims(res[2], 0)

            batch_data = torch.from_numpy(scale(batch_data, factor=scale_factor)).to(device)
            
            with torch.no_grad():
                fake_label = netG(batch_data)
            fake_label = fake_label.to('cpu').numpy()
            
            new_res_list.append((res[0], res[1], np.squeeze(fake_label)))

        end = timer()
        
        print('Processing time is: %.6f' %(end-start))
        
        
        fake_label = size_reduce(new_res_list, org_size, pad_size)
        
        print('reduced_size', fake_label.shape)
        
        res = fake_label
        
        return res


if __name__ == '__main__':
    
    folder_name = '838542'

    file_names = os.listdir(os.path.join('data/_/', folder_name))
    
    dose = '75'
    
    models = ['l1', 'l2', 'ssim', 'pl1', 'pl2', 'pl3', 'pl4_nompool']

    for case in cases:
        
        file_name = folder_name + '_Mammo_' + case + '_' + dose + '.dcm'

        img_ld = dicom.read_file(os.path.join('data/_', folder_name, file_name)).pixel_array
        print('original', img_ld.shape)

        
        for model in models:
    
            generated_image = test_vary_size(img_ld, 75, 0.75, model)


            generated_image = de_scale(generated_image)

            ds = dicom.dcmread(os.path.join('data/_/', folder_name, file_name))
            ds.PixelData = generated_image
            ds.SeriesNumber = np.random.randint(1000000)


            if not os.path.exists('final_resnet_' + folder_name):
                os.mkdir('final_resnet_' + folder_name)


            dicom.write_file('final_resnet_' + folder_name + '/' + 'DL_' + str.upper(model) + '_' + file_name , ds)


