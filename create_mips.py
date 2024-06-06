import numpy as np
from scipy.ndimage import zoom, rotate
import nibabel as nib
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pickle
import os
import time


def create_mips(image_path, n_mips_per_axis=16):
    start_time = time.time()
    print(image_path)
    data = nib.load(image_path)
    image = data.get_fdata()

    # adjust the aspect ratio
    x_scale, y_scale, z_scale = data.header["pixdim"][1:4]
    min_scale = min(x_scale, y_scale, z_scale)
    x_scale /= min_scale
    y_scale /= min_scale
    z_scale /= min_scale
    image = zoom(image, (x_scale, y_scale, z_scale), order=0)

    x, y, z = image.shape
    mips = {"x": {}, "y": {}, "z": {}}

    for i, angle in enumerate(tqdm(np.linspace(0, 90, num=n_mips_per_axis, endpoint=False))):
        rotated_img = rotate(image, angle, axes=(0, 1), reshape=False, order=0, mode="constant", cval=0.0)
        mip = np.max(rotated_img, axis=0)
        mips["x"][angle] = mip

    for i, angle in enumerate(tqdm(np.linspace(0, 90, num=n_mips_per_axis, endpoint=False))):
        rotated_img = rotate(image, angle, axes=(1, 2), reshape=False, order=0, mode="constant", cval=0.0)
        mip = np.max(rotated_img, axis=1)
        mips["y"][angle] = mip

    for i, angle in enumerate(tqdm(np.linspace(0, 90, num=n_mips_per_axis, endpoint=False))):
        rotated_img = rotate(image, angle, axes=(2, 0), reshape=False, order=0, mode="constant", cval=0.0)
        mip = np.max(rotated_img, axis=2)
        mips["z"][angle] = mip

    image_id = image_path.removeprefix("nnUNet_raw/Dataset600_PET_CT_Recon/labelsTs/").removesuffix(".nii.gz")
    for axis in mips:
        for angle in mips[axis]:
            mip = mips[axis][angle]
            out_path = "ground_truth_mips/" + image_id + "_" + axis + "_" + str(round(angle, 6)) + ".pkl"
            with open(out_path, "wb") as f:
                pickle.dump(mip, f)
    total_time = time.time() - start_time
    # save the time taken to create the MIPs
    with open("ground_truth_mips_time/" + image_id + ".txt", "w") as f:
        f.write(str(total_time))


paths = os.listdir("nnUNet_raw/Dataset600_PET_CT_Recon/labelsTs")
image_paths = ["nnUNet_raw/Dataset600_PET_CT_Recon/labelsTs/" + path for path in paths]

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(create_mips, image_path) for image_path in image_paths]
    for future in tqdm(as_completed(futures), total=len(futures)):
        future.result()
