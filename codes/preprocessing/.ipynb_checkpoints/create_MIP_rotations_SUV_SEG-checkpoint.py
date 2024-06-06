#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage import zoom, rotate
from tqdm import tqdm
import os
from os.path import join, dirname, basename, exists
from pprint import pprint
import pickle


# In[ ]:


# segmentation
data_dir = '../../data'

n_mips_per_axis = 4  #should be exponent of 2


# # functions

# In[ ]:


def visualize_slice(img, slice_number):
    plt.imshow(rotate(img[slice_number,:,:], 90, axes=(0,1), reshape=False, order=0, mode='nearest'),
               cmap=cmap)
    plt.show()
    plt.imshow(rotate(img[:,slice_number,:], 90, axes=(0,1), reshape=False, order=0, mode='nearest'),
               cmap=cmap)
    plt.show()
    plt.imshow(rotate(img[:,:,slice_number], 90, axes=(0,1), reshape=False, order=0, mode='nearest'),
               cmap=cmap)
    plt.show()


def dice_score(gt, pred):
    epsilon = 1e-5
    numerator = 2 * ((gt * pred).sum())
    denominator = gt.sum() + pred.sum() + epsilon
    dice = numerator / denominator
    return dice


# # main codes

# In[ ]:


SUV_paths = []
SEG_paths = []

for r,d,f in os.walk(data_dir):
    for filename in f:
        if filename == 'SUV.nii.gz':
            SUV_paths.append(join(r,filename))
        if filename == 'SEG.nii.gz':
            SEG_paths.append(join(r,filename))
            
print(len(SUV_paths))
pprint(SUV_paths[:2])
print(len(SEG_paths))
pprint(SEG_paths[:2])


# In[ ]:


def create_mips(paths):
    for file_path in tqdm(paths):
        filename = basename(file_path)
        data = nib.load(file_path)
        img = data.get_fdata()

        # x is horizontal dimension when view cross sectional image
        # y is vertical dimension when view cross sectional image
        # z is height dimension of the patient
        x,y,z = img.shape
        x_scale, y_scale, z_scale = data.header['pixdim'][1:4]
        min_scale = min(x_scale, y_scale, z_scale)
        x_scale /= min_scale
        y_scale /= min_scale
        z_scale /= min_scale

        img = zoom(img, (x_scale, y_scale, z_scale), order=0)  
        #Spline order 0 is equivalent to nearest neighbor interpolation.

        ###########################
        #create Maximum Intensity Projections (MIPs)
        ###########################

        x,y,z = img.shape

        mips = {'x':{}, 'y':{}, 'z':{}}

        #rotate on the x,y plane, and take MIP along x axis
        for i,angle in enumerate(np.linspace(0, 90, num=n_mips_per_axis, endpoint=False)):
            rotated_img = rotate(img, angle, axes=(0,1), reshape=False,
                                order=0, mode='constant', cval=0.0)
            mip = np.max(rotated_img, axis=0)
            mips['x'][angle] = mip

        #rotate on the y,z plane, and take MIP along y axis
        for i,angle in enumerate(np.linspace(0, 90, num=n_mips_per_axis, endpoint=False)):
            rotated_img = rotate(img, angle, axes=(1,2), reshape=False,
                                order=0, mode='constant', cval=0.0)
            mip = np.max(rotated_img, axis=1)
            mips['y'][angle] = mip

        #rotate on the z,x plane, and take MIP along z axis
        for i,angle in enumerate(np.linspace(0, 90, num=n_mips_per_axis, endpoint=False)):
            rotated_img = rotate(img, angle, axes=(2,0), reshape=False,
                                order=0, mode='constant', cval=0.0)
            mip = np.max(rotated_img, axis=2)
            mips['z'][angle] = mip
        
        if filename == 'SUV.nii.gz':
            mip_dir = join(dirname(file_path).replace('/data/', '/MIP/'), 'SUV_MIP')
            mip_prefix = 'SUV_MIP'
        elif filename == 'SEG.nii.gz':
            mip_dir = join(dirname(file_path).replace('/data/', '/MIP/'), 'SEG_MIP')
            mip_prefix = 'SEG_MIP'
            
        if not exists(mip_dir):
            os.makedirs(mip_dir)
        
        for axis_name in ['x','y','z']:
            for angle in mips[axis_name].keys():
                mip = mips[axis_name][angle]
                out_path = join(mip_dir, mip_prefix + '_' + axis_name + '_' + str(round(angle,2)) + '_' + 'degrees' + '.pkl')
                with open(out_path, 'wb') as f:
                    pickle.dump(mip, f)
            


# In[ ]:


create_mips(SUV_paths)
create_mips(SEG_paths)


# In[ ]:




