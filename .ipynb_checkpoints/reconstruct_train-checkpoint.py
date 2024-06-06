import numpy as np
import os
import nibabel as nib
import scipy.ndimage
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pprint import pprint

# name = "Dataset500_MIP_Segmentation_train"
name = "Dataset501_Synthetic_MIP_Segmentation_train"
dir_path = f"nnUNet_outputs/{name}"
output_dir = f"nnUNet_outputs_reconstructed/{name}"
os.makedirs(output_dir, exist_ok=True)
images = [img for img in os.listdir(dir_path) if img.endswith(".nii.gz")]

def crop(image, new_shape):
    max_dim = max(new_shape)
    scale_factor = max_dim / 512.0

    resized_back_img = scipy.ndimage.zoom(
        image,
        scale_factor,
        order=0,
        mode="constant",
        cval=0.0,
    )

    h, w = resized_back_img.shape
    margin_h = int((h - new_shape[0]) / 2)
    margin_w = int((w - new_shape[1]) / 2)
    # print(resized_back_img.shape)
    # print(margin_h, margin_w, new_shape[0], new_shape[1])
    cropped_img = resized_back_img[margin_h : margin_h + new_shape[0], margin_w : margin_w + new_shape[1]]

    return cropped_img

def dice_score(label, prediction):
    # Flatten the arrays to 1D
    label = label.flatten()
    prediction = prediction.flatten()

    # Calculate intersection and sum of elements
    intersection = np.sum(label * prediction)
    sum_label = np.sum(label)
    sum_prediction = np.sum(prediction)

    if sum_label + sum_prediction == 0:
        return 1

    # Calculate Dice score
    dice = (2.0 * intersection) / (sum_label + sum_prediction)

    return dice

def process_images(images):
    image_split = images[0].removesuffix(".nii.gz").split("_")
    image_id = "_".join(image_split[:3])
    print(image_id, os.path.exists(f"{output_dir}/{image_id}.nii.gz"))
    if os.path.exists(f"{output_dir}/{image_id}.nii.gz"):
        return
    original_suv_path = f"nnUNet_raw/Dataset600_PET_CT_Recon/imagesTr/{image_split[0]}_{image_split[1]}_{image_split[2]}_0000.nii.gz"
    assert os.path.exists(original_suv_path) 
    suv_img = nib.load(original_suv_path)
    image_shape = suv_img.get_fdata().shape

    # original_seg_path = f"nnUNet_raw/Dataset600_PET_CT_Recon/labelsTr/{image_split[0]}_{image_split[1]}_{image_split[2]}.nii.gz"
    # assert os.path.exists(original_seg_path)
    # seg_img = nib.load(original_seg_path)

    # print(suv_img.header)
    x_scale, y_scale, z_scale = suv_img.header['pixdim'][1:4]
    min_scale = min(x_scale, y_scale, z_scale)
    x_scale /= min_scale
    y_scale /= min_scale
    z_scale /= min_scale

    # print(x_scale, y_scale, z_scale)

    image_shape = [
        round(image_shape[0] * x_scale),
        round(image_shape[1] * y_scale),
        round(image_shape[2] * z_scale),
    ]
    segmentation = np.ones(image_shape)

    for image in images:
        image_split = image.removesuffix(".nii.gz").split("_")
        image_id = "_".join(image_split[:3])
        axis = image_split[3]
        angle = float(image_split[4])
        mip = nib.load(f"{dir_path}/{image}").get_fdata()

        if axis == "x":
            mip = crop(mip, (image_shape[1], image_shape[2]))
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
            mip = crop(mip, (image_shape[0], image_shape[2]))
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
            mip = crop(mip, (image_shape[0], image_shape[1]))
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
    
    # print("suv_img.get_fdata().shape =", suv_img.get_fdata().shape)
    # print("revert_segmentation.shape =", revert_segmentation.shape)

    print("done!", image_id)
    # print(dice_score(seg_img.get_fdata(), revert_segmentation.get_fdata()), image_id)
    
    new_img = nib.Nifti1Image(revert_segmentation.astype(np.uint8), suv_img.affine, header=suv_img.header)
    new_img.header.set_data_dtype(np.uint8)
    new_img.header['bitpix'] = 8
    nib.save(new_img, f"{output_dir}/{image_id}.nii.gz")
    return image_id

images = sorted(images)
image_groups = [images[i : i + 12] for i in range(0, len(images), 12)]
print(len(image_groups))
pprint(image_groups[0])

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_images, image_group) for image_group in image_groups]
    for future in tqdm(as_completed(futures), total=len(futures)):
        future.result()