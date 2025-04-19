'''
author: Rupesh Garsondiya
github: @Rupeshgarsondiya
organization: L.J University
'''

import os

class Config():

    def __init__(self):

        # training and validation dataset path before storing dataset into .npy format
        self.CWD = os.getcwd() # set current working directory 
        self.TRAINSET_PATH = os.path.join(self.CWD,'data/raw/files/20k_train') # set training directory path
        self.VALIDSET_PATH = os.path.join(self.CWD,'data/raw/files/real_images') # set validation directory path
        self.META_DATA = os.path.join(self.CWD,'data/raw/train_sample.csv')

        # training and validation dataset path after storing dataset into .npy format
        self.TRAIN_IMAGES_DIR = '' # automatically set path for the directory contains stacked images in .npy format (training)
        self.TRAIN_MASKS_DIR = '' # automatically set path for the directory contains mask images in .npy format (training)
        self.VAL_IMAGES_DIR = '' # set path for the directory contains stacked images in .npy format (validation)
        self.VAL_MASKS_DIR = '' # set path for the directory contains mask images in .npy format (validation)
        self.PATH_TO_SAVE_TRAINED_MODEL = os.path.join(self.CWD,'saved_models/') # set path to save trained model

        # model training parameters
        self.BATCH_SIZE = 16 # set batch size for model training
        self.MAX_EPOCHS = 70 # set maximum epochs for model training
        self.LEARNING_RATE =5e-5 # set learning rate
        self.TRANSFORM = True # set boolean value for applying augmentation techniques for training set and techniques are horizontal flip and vertical flip

    def printConfiguration(self):
        """
        This function is used to print all configuration related to paths and model training params.

        Parameters:
        - (None)

        Returns:
        - (None)
        """
        print('-'*20)
        print(f"Configurations:")
        print('-'*20)
        print(f"Current Working Directory: {self.CWD},\nTrainset_path: {self.TRAINSET_PATH},\nValidset_path: {self.VALIDSET_PATH},\n"
            f"Train_image_path: {self.TRAIN_IMAGES_DIR},\nPath_to_save_processed_data: {self.PATH_TO_SAVE_TRAINED_MODEL},\n"
            f"Path_to_save_trained_model: {self.PATH_TO_SAVE_TRAINED_MODEL},\nBatch_size: {self.BATCH_SIZE},\nMax_epochs: {self.MAX_EPOCHS},\n"
            f"Learning_rate: {self.LEARNING_RATE},\nTransform/Data_augmentation: {self.TRANSFORM}")

        