#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pydicom as dicom
import os
import h5py
import matplotlib.pyplot as plt


path2read = ''

folder_names = sorted(os.listdir(path2read))

#training top 100

np.random.seed(0)

views = ['L_CC', 'L_MLO', 'R_CC', 'R_MLO']


noise_level = 50

def crop_image(img):
    # for normal dose
    mask = img < 7000
    
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
    
    return w_min, h_min, w_max, h_max


def extract_patch(img_ld, img_nd):
    assert img_ld.shape == img_nd.shape, "image sizes differ"
    
    
    w_min, h_min, w_max, h_max = crop_image(img_nd)
    cropped_img_nd = img_nd[w_min:w_max, h_min:h_max]
    cropped_img_ld = img_ld[w_min:w_max, h_min:h_max]
    w, h = cropped_img_nd.shape
    patches = []
    
    for i in range(0, w-64, 64):
        for j in range(0, h-64, 64):
            patch = (cropped_img_ld[i:i+64, j:j+64], cropped_img_nd[i:i+64, j:j+64])
            
            if np.sum(patch[1]>7000) < 64*64:
                patches.append(patch)
    return patches

def process_each_folder(folder_name):
    file_names = os.listdir(os.path.join(path2read, folder_name))
    folder_views = [x for x in views if x in " ".join(file_names)]
    
    patches = []
    for view in folder_views:
        nd_file_name = folder_name + '_Mammo_' + view + '.dcm'
        ld_file_name = folder_name + '_Mammo_' + view + '_' + str(noise_level) +'.dcm'
        
        img_ld = dicom.read_file(os.path.join(path2read, folder_name, ld_file_name)).pixel_array
        img_nd = dicom.read_file(os.path.join(path2read, folder_name, nd_file_name)).pixel_array
        patches += extract_patch(img_ld, img_nd)
        
    return patches
    
    
    
    
if __name__ == '__main__':
    
    patches = []
    for folder_name in folder_names[:100]:
        patches += process_each_folder(folder_name)
    
    rand_index = np.random.permutation(len(patches))
    
    sel_patches = [patches[i] for i in rand_index[:256000]]
    
    data = np.stack([x[0] for x in sel_patches])
    label = np.stack([x[1] for x in sel_patches])
    
    f = h5py.File('DM_training_{}.h5'.format(noise_level), 'w')
    f['data'] = data
    f['label'] = label
    f.close()




