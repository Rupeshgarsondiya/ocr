'''
author: Rupesh Garsondiya
github: @Rupeshgarsondiya
organization: L.J University
'''

import os
import torch
import ocrdata 
import pytorch_lightning as pl

from nn_arch.trocr import TrOcr
from config.config import Config
from ocrdata import OCRDataModule
from gpu_config.check import check_gpu

check_gpu() # This will check if the GPU is available and print the result
One Dark Pr
print('meta data : ',config.META_DATA)
print('image path',config.TRAINSET_PATH)

data_module = OCRDataModule(config.META_DATA, config.TRAINSET_PATH, config.TRANSFORM, config.BATCH_SIZE) # create data module
TROCR_model = TrOcr(config.LEARNING_RATE) # create model

trainer = pl.Trainer(max_epochs=config.MAX_EPOCHS, accelerator="gpu" if torch.cuda.is_available() else "cpu") # create trainer

trainer.fit(TROCR_model, data_module) # train the model

# Define full file path for the model
save_model_path_with_name = os.path.join(config.PATH_TO_SAVE_TRAINED_MODEL,"tr0cr_1.pth") # save the model

# Save the model
torch.save(TROCR_model.state_dict(), save_model_path_with_name)

print(f"Model saved successfully at {save_model_path_with_name}")







