import numpy as np

def dice_score(gt, pred):
    intersection = np.sum((gt > 0) & (pred > 0))
    size_gt = np.sum(gt > 0)
    size_pred = np.sum(pred > 0)
    if size_gt + size_pred == 0:
        return 1.0
    return 2.0 * intersection / (size_gt + size_pred)

def calculate_fp_fn(gt, pred):
    fp = np.sum((gt == 0) & (pred == 1))
    fn = np.sum((gt == 1) & (pred == 0))
    return fp, fn

import os

# dir_path = "nnUNet_outputs/Dataset221_AutoPETII_2023_test"
# dir_path = "nnUNet_outputs_reconstructed/Dataset500_MIP_Segmentation_test"
# dir_path = "nnUNet_outputs_reconstructed/Dataset501_Synthetic_MIP_Segmentation_test"
# dir_path = "nnUNet_outputs/Dataset700_PET_CT_Recon_test"
# dir_path = "nnUNet_outputs/Dataset800_PET_CT_TrueRecon_test"
dir_path = "nnUNet_outputs_reconstructed/ground_truth_mips_4"
samples = os.listdir(dir_path)
samples = [file for file in samples if file.endswith(".nii.gz")]

import nibabel as nib
from tqdm import tqdm

records = []
for sample in tqdm(samples):
    label = nib.load(f"nnUNet_raw/Dataset600_PET_CT_Recon/labelsTs/{sample}")
    # prediction = nib.load(f"{dir_path}/{sample}")
    prediction_1 = nib.load(f"nnUNet_outputs/Dataset221_AutoPETII_2023_test/{sample}")
    # prediction_2 = nib.load(f"nnUNet_outputs_reconstructed/Dataset500_MIP_Segmentation_test/{sample}")
    prediction_2 = nib.load(f"nnUNet_outputs/Dataset700_PET_CT_Recon_test/{sample}")
    # assert label.shape == prediction.shape
    image_label = np.array(label.dataobj)
    image_prediction = np.array(prediction_1.dataobj) * np.array(prediction_2.dataobj)
    pixdim = label.header['pixdim']
    voxel_volume = pixdim[1] * pixdim[2] * pixdim[3] / 1000
    score = dice_score(image_label, image_prediction)
    fp, fn = calculate_fp_fn(image_label, image_prediction)
    # print(fp, fn, pixdim)
    fpv, fnv = fp * voxel_volume, fn * voxel_volume
    records.append({
        "sample": sample,
        "dice_score": score,
        "fpv": fpv,
        "fnv": fnv,
        "label_sum": image_label.sum(),
        "prediction_sum": image_prediction.sum(),
        "voxel_volume": voxel_volume,
    })
    print(records[-1])

import pandas as pd

df = pd.DataFrame(records).set_index("sample")

print(df[df["prediction_sum"] == 0].mean()["dice_score"])
print(df[df["prediction_sum"] != 0].mean()["dice_score"])

print("-" * 10)

print("Negative:", df[df["label_sum"] == 0].mean()["dice_score"])
print("Positive:", df[df["label_sum"] != 0].mean()["dice_score"])

print("-" * 10)

print("Total:", df.mean()["dice_score"])
# df.to_csv("dice_scores/segmentation_221.csv")
# df.to_csv("dice_scores/reconstructed_500.csv")
# df.to_csv("dice_scores/reconstructed_501.csv")
# df.to_csv("dice_scores/segmentation_700.csv")
# df.to_csv("dice_scores/segmentation_800.csv")
# df.to_csv("dice_scores/segmentation_221_700.csv")
df.to_csv("dice_scores/reconstructed_r_4.csv")