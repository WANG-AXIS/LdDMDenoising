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
from collections import namedtuple



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


##################Custom setting###############


DIM = 64


BATCH_SIZE =  256

LR_g = 0.0001
LR_d = 0.0001
Model_folder = 'final_models/'
Log_folder = "final_logs"


noise_level = 75
scale_factor = 0.75

###############################################

if not os.path.exists('final_models'):
    os.mkdir('final_models')

if not os.path.exists('final_logs'):
    os.mkdir('final_logs')
    
    
appendix = 'pl1_{}'.format(noise_level)
    
###############################################


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = torch.cat([X, X, X], dim=1)
        #X = normalize_batch(X)
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out



class Vgg16_NoMaxPooling(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16_NoMaxPooling, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice = torch.nn.Sequential()
        for x in range(23):
            if x not in [4, 9, 16]: # remove max pooling
                self.slice.add_module(str(x), vgg_pretrained_features[x])
    
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = torch.cat([X, X, X], dim=1)

        h = self.slice(X) # PL4 without max pooling

        return h

    
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


vgg = Vgg16(requires_grad=False).to(device)
#vgg_nomaxpooling = Vgg16_NoMaxPooling(requires_grad=False).to(device)
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
        
        features_y = vgg(fake_label)
        features_x = vgg(batch_label)
            
        pl1_loss =   F.mse_loss(features_x.relu1_2, features_y.relu1_2)
        pl2_loss =   F.mse_loss(features_x.relu2_2, features_y.relu2_2)
        pl3_loss =   F.mse_loss(features_x.relu3_3, features_y.relu3_3)
        pl4_loss =   F.mse_loss(features_x.relu4_3, features_y.relu4_3)
        #pl4_nomaxpooling_loss = F.mse_loss(vgg_nomaxpooling(fake_label), vgg_nomaxpooling(batch_label))
            
        gen_cost = pl1_loss

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
            pl1_loss.item(),
            pl2_loss.item(),
            pl3_loss.item(),
            pl4_loss.item(),
           
        ))


if __name__ == '__main__':
    for epoch in range(60):
        train(tr_data, tr_label, epoch)   
        scheduler_g.step()
        torch.save(netG.state_dict(), Model_folder + 'model_ResResNet_{}.pth'.format(appendix))
        
    os._exit(00)
 



