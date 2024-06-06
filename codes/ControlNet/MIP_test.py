import pytorch_lightning as pl
from torch.utils.data import DataLoader
from MIP_dataset import MIPDataset
from cldm.logger import MIPImageLogger

from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import pickle
from os.path import join, dirname, exists
import os
import pandas as pd
from tqdm import tqdm
import shutil

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.uniformer import UniformerDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


dataset_path = '../../spreadsheet/MIP_paths_and_prompts_square512.xlsx'
export_path = '../../spreadsheet/MIP_paths_and_prompts_square512_gen.xlsx'

apply_uniformer = UniformerDetector()
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./lightning_logs/version_2/checkpoints/epoch=9-step=24359.ckpt', location='cuda'))
model.sd_locked = True
model.only_mid_control = False
model = model.cuda()
ddim_sampler = DDIMSampler(model)

#print(vars(model))


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    
    with torch.no_grad():
        #input_image = HWC3(input_image)  #no need
        #detected_map = apply_uniformer(resize_image(input_image, detect_resolution))  #no need
        
        detected_map = input_image
        #img = resize_image(input_image, image_resolution)  #no need
        #H, W, C = img.shape
        H, W, C = (512, 512, 3)

        #detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

        #control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.from_numpy(detected_map.copy()).float().cuda()  #The mask is already within range [0,1]
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        #cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        #x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        x_samples = ((einops.rearrange(x_samples, 'b c h w -> b h w c')+1.0)*250.0).cpu().numpy().clip(0,None)  #because we did (x/250)-1 during training
        #convert to RGB to GRAY (cannot use opencv because the range is not [0,255])
        x_samples = (0.299 * x_samples[:,:,:,0]) + (0.587 * x_samples[:,:,:,1]) + (0.114 * x_samples[:,:,:,2])
        
        results = [x_samples[i] for i in range(num_samples)]
        
    return results  #[detected_map] + results


#####################
#main codes
#####################

num_samples = 1
image_resolution = 512
strength = 1.0  #"Control Strength", minimum=0.0, maximum=2.0, value=1.0
guess_mode = False  #True if no text prompt
detect_resolution = 512  #"Segmentation Resolution", minimum=128, maximum=1024
ddim_steps = 100  #"Steps", minimum=1, maximum=100, value=20
scale = 9.0  #"Guidance Scale", minimum=0.1, maximum=30.0
seed = 0  #"Seed", minimum=-1, maximum=2147483647, randomize=True
eta = 0.0  #"eta (DDIM)"
a_prompt = ""  #"Added Prompt", value='best quality, extremely detailed'
n_prompt = ""  #"Negative Prompt", value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

df = pd.read_excel(dataset_path)

for row_idx in tqdm(df.index):
    seed = row_idx
    seg_path = df.loc[row_idx,'SEG_MIP_path']
    suv_path = df.loc[row_idx,'SUV_MIP_path']
    prompt = df.loc[row_idx,'prompt']
    
    with open(seg_path, 'rb') as f:
        input_image = pickle.load(f)  #segmentation mask
    height,width = input_image.shape
    input_image = np.broadcast_to(input_image[:,:,np.newaxis], (height,width,3))
    
    gen_images = process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)
    
    gen_seg_path = seg_path.replace('/MIP_square512/', '/gen_MIP_square512/')
    gen_suv_path = suv_path.replace('/MIP_square512/', '/gen_MIP_square512/')
    df.loc[row_idx, 'gen_SUV_MIP_path'] = gen_suv_path
    df.loc[row_idx, 'gen_SEG_MIP_path'] = gen_seg_path
    
    gen_seg_dir = dirname(gen_seg_path)
    if not exists(gen_seg_dir):
        os.makedirs(gen_seg_dir)
    gen_suv_dir = dirname(gen_suv_path)
    if not exists(gen_suv_dir):
        os.makedirs(gen_suv_dir)

    for i,img in enumerate(gen_images):
        with open(gen_suv_path, 'wb') as f:
            pickle.dump(img, f)
    
    shutil.copy(seg_path, gen_seg_path)
    
    if row_idx % 10 == 0:
        df.to_excel(export_path)
    

    
    
    


