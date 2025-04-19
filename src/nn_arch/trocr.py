'''
author: Rupesh Garsondiya
github: @Rupeshgarsondiya
organization: L.J University
'''

import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from transformers import VisionEncoderDecoderModel

class TrOcr(pl.LightningModule) :
    def __init__(self, learning_rate, model_name="microsoft/trocr-small-printed"):
        super().__init__()
        self.learning_rate = learning_rate  # Set learning rate
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

        # Ensure pad_token_id is set
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id

        # Fix: Set decoder_start_token_id manually
        if self.model.config.decoder_start_token_id is None:
            self.model.config.decoder_start_token_id = self.model.config.pad_token_id  # Set to `pad_token_id`

        # Print to verify
        print(f"Pad Token ID: {self.model.config.pad_token_id}")
        print(f"Decoder Start Token ID: {self.model.config.decoder_start_token_id}")

    def forward(self, pixel_values, labels=None) -> torch.Tensor:
        '''
        Args:
        - pixel_values (torch.Tensor): Input image tensor

        Returns:
        - logits (torch.Tensor): Output tensor
        '''
        return self.model(pixel_values=pixel_values, labels=labels)  # Correct!

    def training_step(self, batch, batch_idx) -> dict:
        '''
        Args:
        - batch (dict): Training batch

        Returns:
        - loss (torch.Tensor): Training loss
        '''
        images, labels = batch  # get the images and its labels from the batch data
        loss = torch.nn.CrossEntropyLoss()
        outputs = self.forward(images,labels) # pass images to model architecture or forward pass
        loss = outputs.loss
        # calculate the cross entropy loss
        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        '''
        Returns:
        - optimizer (torch.optim.Adam): Optimizer instance
        '''
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    


    def generate(self,*args,**kwargs) -> dict:
        '''
        Args:
        - args (list): List of arguments
        - kwargs (dict): Dictionary of keyword arguments
        '''
        return self.model.generate(*args,**kwargs)
        