import numpy as np
import nibabel as nib
import pathlib as plb
import cc3d
import csv
import sys
import pandas as pd
from tqdm import tqdm
import os


def nii2numpy(nii_path):
    # input: path of NIfTI segmentation file, output: corresponding numpy array and voxel_vol in ml
    mask_nii = nib.load(str(nii_path))
    mask = mask_nii.get_fdata()
    pixdim = mask_nii.header['pixdim']   
    voxel_vol = pixdim[1]*pixdim[2]*pixdim[3]/1000
    return mask, voxel_vol


def con_comp(seg_array):
    # input: a binary segmentation array output: an array with seperated (indexed) connected components of the segmentation array
    connectivity = 18
    conn_comp = cc3d.connected_components(seg_array, connectivity=connectivity)
    return conn_comp


def false_pos_pix(gt_array,pred_array):
    # compute number of voxels of false positive connected components in prediction mask
    pred_conn_comp = con_comp(pred_array)
    
    false_pos = 0
    for idx in range(1,pred_conn_comp.max()+1):
        comp_mask = np.isin(pred_conn_comp, idx)
        if (comp_mask*gt_array).sum() == 0:
            false_pos = false_pos+comp_mask.sum()
    return false_pos



def false_neg_pix(gt_array,pred_array):
    # compute number of voxels of false negative connected components (of the ground truth mask) in the prediction mask
    gt_conn_comp = con_comp(gt_array)
    
    false_neg = 0
    for idx in range(1,gt_conn_comp.max()+1):
        comp_mask = np.isin(gt_conn_comp, idx)
        if (comp_mask*pred_array).sum() == 0:
            false_neg = false_neg+comp_mask.sum()
            
    return false_neg


def dice_score(mask1,mask2):
    # compute foreground Dice coefficient
    overlap = (mask1*mask2).sum()
    sum = mask1.sum()+mask2.sum()
    dice_score = 2*overlap/sum
    return dice_score



def compute_metrics(nii_gt_path, nii_pred_path):
    # main function
    gt_array, voxel_vol = nii2numpy(nii_gt_path)
    pred_array, voxel_vol = nii2numpy(nii_pred_path)

    false_neg_vol = false_neg_pix(gt_array, pred_array)*voxel_vol
    false_pos_vol = false_pos_pix(gt_array, pred_array)*voxel_vol
    dice_sc = dice_score(gt_array,pred_array)

    return dice_sc, false_pos_vol, false_neg_vol


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python eval.py <nii_pred_folder_path> <dataset_ID>")
        sys.exit(1)
    file_path = "data_split_study_level.xlsx"
    df = pd.read_excel(file_path)
    sim_df = df[["Subject ID","Study ID", "diagnosis","allocated_set"]]
    sim_df = sim_df[sim_df["allocated_set"] == "test"]
    nii_pred_folder_path = sys.argv[1]
    nii_gt_folder_path = "/nobackup2/jirayu/data/data/"
    dataset_ID = sys.argv[2]
        
    # nii_pred_folder_path = "/nobackup-fast/jirayu/nnUNet/nnUNet_results/Dataset221_AutoPETII_2023/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation"
    nii_gt_folder_path = "/nobackup2/jirayu/data/data/"
    # Open the CSV file in write mode and write the header once
    with open(f"Dataset{dataset_ID}_metrics.csv", "w", newline='') as f:
        if sim_df.empty:
            print("The similarity DataFrame is empty. Please check the input file.")
            sys.exit(1)
    
        csv_header = ['gt_name', 'dice_sc', 'false_pos_vol', 'false_neg_vol', 'diagnosis']
        writer = csv.writer(f, delimiter=',')
        writer.writerow(csv_header)
        for index, row in tqdm(sim_df.iterrows()):
            patient_ID = row["Subject ID"]
            study_ID = row["Study ID"]
            diagnosis = row["diagnosis"]
            pred_filename = f"{patient_ID}_{study_ID}.nii.gz"
            pred_path = os.path.join(nii_pred_folder_path, pred_filename)
            gt_path = os.path.join(nii_gt_folder_path, patient_ID, study_ID, "SEG.nii.gz")
    
            if os.path.exists(pred_path) and os.path.exists(gt_path):
                try:
                    dice_sc, false_pos_vol, false_neg_vol = compute_metrics(gt_path, pred_path)
                    print(dice_sc, false_pos_vol,false_neg_vol)
                    csv_row = [os.path.join(patient_ID, study_ID), dice_sc, false_pos_vol, false_neg_vol, diagnosis]
                    writer.writerow(csv_row)
                except Exception as e:
                    print(f"Error processing {pred_filename}: {e}")
            else:
                    print(f"Prediction or ground truth file not found for {patient_ID}_{study_ID}")
            # dice_sc, false_pos_vol, false_neg_vol = compute_metrics(gt_path, pred_path)
            
            # csv_header = ['gt_name', 'dice_sc', 'false_pos_vol', 'false_neg_vol', 'diagnosis']
            # csv_row = [os.path.join(patient_ID,study_ID), dice_sc, false_pos_vol, false_neg_vol,diagnosis]
            # writer.writerow(csv_row)
            # writer = csv.writer(f, delimiter=',')
            # writer.writerow(csv_header) 
            # writer.writerows(csv_rows)
             # Open the CSV file in append mode and write the row

    
    
    
    '''nii_pred_folder_path, dataset_ID = sys.argv
    # nii_pred_folder_path = "/nobackup-fast/jirayu/nnUNet/nnUNet_results/Dataset221_AutoPETII_2023/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation"
    nii_gt_folder_path = "/nobackup2/jirayu/data/data/"
# Open the CSV file in write mode and write the header once
    with open(f"Dataset{dataset_ID}_metrics.csv", "w", newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(csv_header)
        for filename in os.listdir(nii_pred_folder_path):
            splits = filename.split("_")
            # print(filename.split("_"))
        
            patient_ID = splits[0] + "_" + splits[1]
            study_ID = splits[2][:-7]
            study_ID_condition = sim_df["Study ID"] == study_ID
            patient_ID_condition = sim_df["Subject ID"] == patient_ID
            diagnosis = sim_df[(study_ID_condition) & (patient_ID_condition)]["diagnosis"].values[0]
            pred_path = os.path.join(nii_pred_folder_path, filename)
            gt_path = os.path.join(nii_gt_folder_path, patient_ID, study_ID, "SEG.nii.gz")
            dice_sc, false_pos_vol, false_neg_vol = compute_metrics(gt_path, pred_path)
            
            csv_header = ['gt_name', 'dice_sc', 'false_pos_vol', 'false_neg_vol', 'diagnosis']
            csv_row = [os.path.join(patient_ID,study_ID), dice_sc, false_pos_vol, false_neg_vol,diagnosis]
            # writer = csv.writer(f, delimiter=',')
            # writer.writerow(csv_header) 
            # writer.writerows(csv_rows)
             # Open the CSV file in append mode and write the row
            with open("221_metrics.csv", "a", newline='') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(csv_row)
    
    # break'''




