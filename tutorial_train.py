from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader # Data_iterate
from tutorial_dataset import MyDataset # 创建一个DataLoader
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

# ----------------------------------------
# step 1: Configs
resume_path = './models/control_sd15_ini.ckpt'
batch_size = 2
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# ----------------------------------------
# step 2 : model
# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# ----------------------------------------
# step 3 : data
# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])


# Train!
# if __name__ =="__main__":
#     trainer.fit(model, dataloader)
trainer.fit(model, dataloader)
