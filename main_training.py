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


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


##################Custom setting###############


DIM = 64


BATCH_SIZE =  256

LR_g = 0.0001
LR_d = 0.0001
Model_folder = 'final_models/'
Log_folder = "final_logs"


noise_level = 75 # 25, 50, 75
scale_factor = 0.75 # 0.25, 0.5, 0.75

###############################################

if not os.path.exists('final_models'):
    os.mkdir('final_models')

if not os.path.exists('final_logs'):
    os.mkdir('final_logs')
    
# l1, l2, ssim
appendix = 'ssim_{}'.format(noise_level)
    
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


optimizer_g = optim.Adam(netG.parameters(), lr=LR_g, betas=(0.5, 0.999))
scheduler_g = optim.lr_scheduler.MultiStepLR(optimizer_g, milestones=[10, 20, 30, 40, 50], gamma=0.5)



def scale(x, vmin=0, vmax=8000.0, factor=1.):
    # [-1,1]
    x = x / vmax / factor
    x[x > 1.0] = 1.0
    x[x < 0.0] = 0.0
    x = np.expand_dims(x, axis=1)
    return x.astype(np.float32)



def read_data(file_name):
    f = h5py.File(file_name, 'r')
    data = np.array(f['data'])
    label = np.array(f['label'])
    f.close()
    
    return data, label
    

        
tr_data, tr_label = read_data('DM_training_{}.h5'.format(noise_level)) 


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s, %(message)s',
                    filename= Log_folder + '/log_ResResNet_{}.log'.format(appendix),
                    filemode='w')

logging.info('Epoch, batch, time,  l2_loss, l1_loss, ssim_loss, PL1, PL2, PL3, PL4, is_training')


#adversarial_loss = torch.nn.BCELoss()
ssim_ssim_loss = pytorch_ssim.SSIM(window_size = 11)

def train(data, label, epoch):
    
    data, label = shuffle(data, label)
    for iteration in range(data.shape[0] // BATCH_SIZE):#data.shape[0] // BATCH_SIZE
        
        start_time = time.time()
        start = timer()
        
        batch_data = torch.from_numpy(scale(data[iteration * BATCH_SIZE : (iteration + 1) * BATCH_SIZE], factor=scale_factor)).to(device)
        batch_label = torch.from_numpy(scale(label[iteration * BATCH_SIZE : (iteration + 1) * BATCH_SIZE])).to(device)
        
        
        netG.zero_grad()
        fake_label = netG(batch_data)

        l2_loss = F.mse_loss(fake_label, batch_label) 
        ssim_loss =  ssim_ssim_loss(fake_label, batch_label) 
        l1_loss = F.l1_loss(fake_label, batch_label)
        
        gen_cost = 1 - ssim_loss # l1, l2, ssim

        gen_cost.backward()
        optimizer_g.step()
        
        
        end = timer()
        
        print('Iter: %4d, Loss: %.6f, '%(
            iteration,
            gen_cost.item(),

        ))
        
        logging.info('%d, %d, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, 1'%(
            epoch, 
            iteration, 
            end-start, 
            l2_loss.item(),
            l1_loss.item(),
            ssim_loss.item(),
            0,#pl1_cost.item(), # save training time
            0,#pl2_cost.item(),
            0,#pl3_cost.item(),
            0,#pl4_cost.item(),
           
        ))


if __name__ == '__main__':
    for epoch in range(60):
        train(tr_data, tr_label, epoch)   
        scheduler_g.step()
        torch.save(netG.state_dict(), Model_folder + 'model_ResResNet_{}.pth'.format(appendix))
        
    os._exit(00)
