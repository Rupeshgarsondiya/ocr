'''
author: Rupesh Garsondiya
github: @Rupeshgarsondiya
organization: L.J University
'''

import os
import PIL
import torch
import time
import pandas as pd


from PIL import Image
from db import connect_db
from nn_arch.trocr import TrOcr 
from config.config import Config  
from torchvision import transforms
from transformers import AutoTokenizer
from transformers import TrOCRProcessor

conn = connect_db()
cursor = conn.cursor()

def predict_text(image_path,model):
    '''
    This function takes an image path as input and returns the text extracted from the image.

    Args:
    - image_path (str): The path to the image file.

    Returns:
    - text (str): The text extracted from the image.
    '''
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed",use_fast=False)
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values

    tokenizer = AutoTokenizer.from_pretrained("microsoft/trocr-small-printed")

    # Explicitly provide decoder input IDs
    decoder_input_ids = torch.tensor([[tokenizer.cls_token_id]])

    with torch.no_grad():
        # generated_ids = model(pixel_values, decoder_input_ids=decoder_input_ids)
        generated_ids = model.generate(pixel_values,max_length=2,num_beams=4,repetition_penalty=2.5)

    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return predicted_text

def inference():
    '''
    This function performs the inference on the test dataset.

    Args:
    - None

    Returns:
    - None
    '''

    # Load Configuration
    config = Config()  # Load config settings

    print("-------------------------------------------------")
    print("-------------- LOAD TRAINED MODEL ---------------")
    print("-------------------------------------------------")

    model_path = os.path.join(config.PATH_TO_SAVE_TRAINED_MODEL, "tr0cr_1.pth")  # Update this if needed

    model = TrOcr(config.LEARNING_RATE)
    model.load_state_dict(torch.load(model_path))
    
    print("Model loaded successfully.")

    print("-------------------------------------------------")
    print("------------------ INFERENCE --------------------")
    print("-------------------------------------------------")

    image_path = '/home/rupesh-garsondiya/workstation/lab/OCR/data/raw/files/20k_train/img_19970.jpg' # image path
    
    output_text = predict_text(image_path,model)
    print('Output : ',output_text)


if __name__ == "__main__":
    inference()  # Call the inference function