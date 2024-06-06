import numpy as np
import cc3d
import os
import nibabel as nib
from tqdm import tqdm
import pandas as pd
import argparse

def dice_score(gt, pred):
    intersection = np.sum((gt > 0) & (pred > 0))
    size_gt = np.sum(gt > 0)
    size_pred = np.sum(pred > 0)
    if size_gt + size_pred == 0:
        return 1.0
    return 2.0 * intersection / (size_gt + size_pred)

def con_comp(seg_array):
    connectivity = 18
    conn_comp = cc3d.connected_components(seg_array, connectivity=connectivity)
    return conn_comp

def false_pos_pix(gt_array, pred_array):
    pred_conn_comp = con_comp(pred_array)
    false_pos = 0
    for idx in range(1, pred_conn_comp.max()+1):
        comp_mask = np.isin(pred_conn_comp, idx)
        if (comp_mask * gt_array).sum() == 0:
            false_pos += comp_mask.sum()
    return false_pos

def false_neg_pix(gt_array, pred_array):
    gt_conn_comp = con_comp(gt_array)
    false_neg = 0
    for idx in range(1, gt_conn_comp.max()+1):
        comp_mask = np.isin(gt_conn_comp, idx)
        if (comp_mask * pred_array).sum() == 0:
            false_neg += comp_mask.sum()
    return false_neg

def calculate_fp_fn(gt, pred):
    return false_pos_pix(gt, pred), false_neg_pix(gt, pred)

def main(dataset_number):
    dataset_paths = {
        "221": "nnUNet_outputs/Dataset221_AutoPETII_2023_test",
        "500": "nnUNet_outputs_reconstructed/Dataset500_MIP_Segmentation_test",
        "501": "nnUNet_outputs_reconstructed/Dataset501_Synthetic_MIP_Segmentation_test",
        "700": "nnUNet_outputs/Dataset700_PET_CT_Recon_test",
        "701": "nnUNet_outputs/Dataset701_PET_CT_Recon_Synthetic",
        "800": "nnUNet_outputs/Dataset800_PET_CT_TrueRecon_test",
    }

    dir_path = dataset_paths[str(dataset_number)]
    samples = os.listdir(dir_path)
    samples = [file for file in samples if file.endswith(".nii.gz")]

    records = []
    for sample in tqdm(samples):
        label = nib.load(f"nnUNet_raw/Dataset600_PET_CT_Recon/labelsTs/{sample}")
        prediction = nib.load(f"{dir_path}/{sample}")
        assert label.shape == prediction.shape
        image_label = np.array(label.dataobj)
        image_prediction = np.array(prediction.dataobj)
        pixdim = label.header['pixdim']
        voxel_volume = pixdim[1] * pixdim[2] * pixdim[3] / 1000
        score = dice_score(image_label, image_prediction)
        fp, fn = calculate_fp_fn(image_label, image_prediction)
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

    df = pd.DataFrame(records).set_index("sample")

    print(df[df["prediction_sum"] == 0].mean()["dice_score"])
    print(df[df["prediction_sum"] != 0].mean()["dice_score"])

    print("-" * 10)

    print("Negative:", df[df["label_sum"] == 0].mean()["dice_score"])
    print("Positive:", df[df["label_sum"] != 0].mean()["dice_score"])

    print("-" * 10)

    print("Total:", df.mean()["dice_score"])

    output_path = f"dice_scores/corrected_segmentation_{dataset_number}.csv"
    df.to_csv(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run segmentation analysis.")
    parser.add_argument("dataset_number", type=int, choices=[221, 500, 501, 700, 701, 800], 
                        help="Dataset number to analyze (221, 500, 501, 700, 701, or 800).")
    args = parser.parse_args()
    main(args.dataset_number)
