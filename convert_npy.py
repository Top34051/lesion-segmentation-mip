import os
import nibabel as nib
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def convert_single_image(image_id, images_dir, labels_dir, output_dir):
    try:
        print(f"Processing {image_id}...")

        pet_img = nib.load(os.path.join(images_dir, f'{image_id}_0000.nii.gz')).get_fdata()
        ct_img = nib.load(os.path.join(images_dir, f'{image_id}_0001.nii.gz')).get_fdata()
        recon_img = nib.load(os.path.join(images_dir, f'{image_id}_0002.nii.gz')).get_fdata()
        label = nib.load(os.path.join(labels_dir, f'{image_id}.nii.gz')).get_fdata()
        
        np.save(os.path.join(output_dir, f'{image_id}_pet.npy'), pet_img)
        np.save(os.path.join(output_dir, f'{image_id}_ct.npy'), ct_img)
        np.save(os.path.join(output_dir, f'{image_id}_recon.npy'), recon_img)
        np.save(os.path.join(output_dir, f'{image_id}_label.npy'), label)

        print(f"Finished processing {image_id}")
    except Exception as e:
        print(f"Error processing {image_id}: {e}")

def convert_nii_to_npy(images_dir, labels_dir, output_dir, num_workers=4):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_ids = ['_'.join(f.split('_')[:3]) for f in os.listdir(images_dir) if f.endswith('_0000.nii.gz')]
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(convert_single_image, image_id, images_dir, labels_dir, output_dir) for image_id in image_ids]
        for future in futures:
            future.result()  # Wait for all futures to complete

# Paths to your dataset
images_dir = 'nnUNet_raw/Dataset700_PET_CT_Recon/imagesTr'
labels_dir = 'nnUNet_raw/Dataset700_PET_CT_Recon/labelsTr'
output_dir = 'Dataset700_PET_CT_Recon_npy'

# Convert the NIfTI images to NPY with parallel processing
convert_nii_to_npy(images_dir, labels_dir, output_dir, num_workers=8)
