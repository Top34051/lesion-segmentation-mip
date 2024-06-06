import os
import nibabel as nib
import shutil
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

dir_path = "nnUNet_raw/Dataset700_PET_CT_Recon/"
os.makedirs(dir_path, exist_ok=True)
os.makedirs(dir_path + "imagesTr", exist_ok=True)
os.makedirs(dir_path + "imagesTs", exist_ok=True)
os.makedirs(dir_path + "labelsTr", exist_ok=True)
os.makedirs(dir_path + "labelsTs", exist_ok=True)

def process_row(row, dir_path):
    split = row["allocated_set"]
    subject_id = row["Subject ID"]
    study_id = row["Study ID"]
    if split == "train_val":
        suv_path = f"nnUNet_raw/Dataset600_PET_CT_Recon/imagesTr/{subject_id}_{study_id}_0000.nii.gz"
        ct_path = f"nnUNet_raw/Dataset600_PET_CT_Recon/imagesTr/{subject_id}_{study_id}_0001.nii.gz"
        seg_path = f"nnUNet_raw/Dataset600_PET_CT_Recon/labelsTr/{subject_id}_{study_id}.nii.gz"
        recon_path = f"nnUNet_outputs_reconstructed/Dataset500_MIP_Segmentation_train/{subject_id}_{study_id}.nii.gz"

        suv_dest_path = os.path.join(dir_path, f"imagesTr/{subject_id}_{study_id}_0000.nii.gz")
        ct_dest_path = os.path.join(dir_path, f"imagesTr/{subject_id}_{study_id}_0001.nii.gz")
        recon_dest_path = os.path.join(dir_path, f"imagesTr/{subject_id}_{study_id}_0002.nii.gz")
        seg_dest_path = os.path.join(dir_path, f"labelsTr/{subject_id}_{study_id}.nii.gz")

    else:
        suv_path = f"nnUNet_raw/Dataset600_PET_CT_Recon/imagesTs/{subject_id}_{study_id}_0000.nii.gz"
        ct_path = f"nnUNet_raw/Dataset600_PET_CT_Recon/imagesTs/{subject_id}_{study_id}_0001.nii.gz"
        seg_path = f"nnUNet_raw/Dataset600_PET_CT_Recon/labelsTs/{subject_id}_{study_id}.nii.gz"
        recon_path = f"nnUNet_outputs_reconstructed/Dataset500_MIP_Segmentation_test/{subject_id}_{study_id}.nii.gz"

        suv_dest_path = os.path.join(dir_path, f"imagesTs/{subject_id}_{study_id}_0000.nii.gz")
        ct_dest_path = os.path.join(dir_path, f"imagesTs/{subject_id}_{study_id}_0001.nii.gz")
        recon_dest_path = os.path.join(dir_path, f"imagesTs/{subject_id}_{study_id}_0002.nii.gz")
        seg_dest_path = os.path.join(dir_path, f"labelsTs/{subject_id}_{study_id}.nii.gz")

    print(suv_path)
    assert os.path.exists(suv_path)
    assert os.path.exists(ct_path)
    assert os.path.exists(recon_path)
    assert os.path.exists(seg_path)

    shutil.copy(suv_path, suv_dest_path)
    shutil.copy(ct_path, ct_dest_path)
    shutil.copy(recon_path, recon_dest_path)
    shutil.copy(seg_path, seg_dest_path)

def main(df, dir_path):
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_row, [row for _, row in df.iterrows()], [dir_path]*len(df)), total=len(df)))

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_excel("data_split_study_level.xlsx")
    main(df, dir_path)
