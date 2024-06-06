import numpy as np
import os
from os.path import join, exists, dirname
import nibabel as nib
import scipy.ndimage
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pprint import pprint
import pandas as pd
import pickle


study_df = pd.read_excel('../../spreadsheet/data_split_study_level.xlsx')
mip_df = pd.read_excel('../../spreadsheet/MIP_paths_and_prompts_square512_gen_complete.xlsx')


def process_image(study_idx):
    subject_id = study_df.loc[study_idx, 'Subject ID']
    study_id = study_df.loc[study_idx, 'Study ID']
    original_seg_path = join('../../data', subject_id, study_id, 'SEG.nii.gz')

    seg_img = nib.load(original_seg_path)

    image_shape = seg_img.get_fdata().shape

    x_scale, y_scale, z_scale = seg_img.header['pixdim'][1:4]
    min_scale = min(x_scale, y_scale, z_scale)
    x_scale /= min_scale
    y_scale /= min_scale
    z_scale /= min_scale

    image_shape = [
        round(image_shape[0] * x_scale),
        round(image_shape[1] * y_scale),
        round(image_shape[2] * z_scale),
    ]
    segmentation = np.ones(image_shape)

    mip_indices = mip_df.loc[mip_df['Study ID']==study_id].index

    for mip_idx in mip_indices:
        mip_path = mip_df.loc[mip_idx, 'SEG_MIP_path'].replace('/MIP_square512/', '/MIP/')
        #print(mip_path)
        axis = mip_df.loc[mip_idx, 'projection_axis']
        angle = mip_df.loc[mip_idx, 'rotation_degrees']
        with open(mip_path, 'rb') as f:
            mip = pickle.load(f)

        if axis == "x":
            #mip = crop(mip, (image_shape[1], image_shape[2]))
            broadcast_mip = np.broadcast_to(mip[np.newaxis, :, :], image_shape)
            reversed_rotate = scipy.ndimage.rotate(
                broadcast_mip,
                360 - angle,
                axes=(0, 1),
                reshape=False,
                order=0,
                mode="constant",
                cval=0,
            )
        elif axis == "y":
            #mip = crop(mip, (image_shape[0], image_shape[2]))
            broadcast_mip = np.broadcast_to(mip[:, np.newaxis, :], image_shape)
            reversed_rotate = scipy.ndimage.rotate(
                broadcast_mip,
                360 - angle,
                axes=(1, 2),
                reshape=False,
                order=0,
                mode="constant",
                cval=0,
            )
        elif axis == "z":
            #mip = crop(mip, (image_shape[0], image_shape[1]))
            broadcast_mip = np.broadcast_to(mip[:, :, np.newaxis], image_shape)
            reversed_rotate = scipy.ndimage.rotate(
                broadcast_mip,
                360 - angle,
                axes=(2, 0),
                reshape=False,
                order=0,
                mode="constant",
                cval=0,
            )
        segmentation *= reversed_rotate

    revert_segmentation = scipy.ndimage.zoom(
        segmentation,
        (1.0 / x_scale, 1.0 / y_scale, 1.0 / z_scale),
        order=0,
    )

    new_img = nib.Nifti1Image(revert_segmentation.astype(np.uint8), seg_img.affine, header=seg_img.header)
    new_img.header.set_data_dtype(np.uint8)
    new_img.header['bitpix'] = 8
    out_path = original_seg_path.replace('/data/', '/reconstructed_3D_GT_segmentation/').replace('SEG.nii.gz', 'recon_SEG.nii.gz')
    
    out_dir = dirname(out_path)
    if not exists(out_dir):
        os.makedirs(out_dir)
    nib.save(new_img, out_path)
    
    print("done!", study_idx, study_id)
        

max_workers = 8
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(process_image, study_idx): study_idx for study_idx in study_df.index}
    for future in tqdm(as_completed(futures)):
        try:
            # If a function completes without errors, it will return its result (if any)
            result = future.result()  
        except Exception as exc:
            # If the function raised an exception, calling `future.result()` will raise the same exception
            print(f'An error occurred: {exc}')
            