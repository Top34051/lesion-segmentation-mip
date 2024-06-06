from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


def main():
    # Configs
    resume_path = './models/control_sd15_ini.ckpt'
    batch_size = 1  #4
    logger_freq = 300
    learning_rate = 1e-5
    sd_locked = True
    only_mid_control = False


    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control


    # Misc
    dataset = MyDataset()
    
    #dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    dataloader = DataLoader(dataset, num_workers=8, batch_size=batch_size, shuffle=True)
    #dataloader = DataLoader(dataset, num_workers=8, batch_size=batch_size, shuffle=True, persistent_workers=True) #for multiple GPUs training
    
    logger = ImageLogger(batch_frequency=logger_freq)
    
    #trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])
    trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger], accumulate_grad_batches=4)  # But this will be 4x slower
    #trainer = pl.Trainer(gpus=2, precision=32, callbacks=[logger], accumulate_grad_batches=4)

    # Train!
    trainer.fit(model, dataloader)

    

if __name__ == '__main__':
    main()


    
#The original codes didn't wrap as a main() function.
"""
# Configs
resume_path = './models/control_sd15_ini.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
#dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
dataloader = DataLoader(dataset, num_workers=8, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
#trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])
#trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger], accumulate_grad_batches=4)  # But this will be 4x slower
trainer = pl.Trainer(gpus=2, precision=32, callbacks=[logger])

# Train!
trainer.fit(model, dataloader)
"""
