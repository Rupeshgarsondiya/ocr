'''
author: Rupesh Garsondiya
github: @Rupeshgarsondiya
organization: L.J University
'''

import torch
import subprocess

def check_gpu()->None:
    '''
    This function checks if the system has a GPU available.

    Args:
        (None)

    Returns:
        (None)
    '''

    if torch.cuda.is_available():

        device_count = torch.cuda.device_count() # get the number of GPUs available
        device= torch.cuda.get_device_name() # get the name of the GPU

        print(f"Total GPU count: {device_count}")

        for i in range(device_count): # loop through each GPU
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}") 
        
        command = "nvidia-smi" # set a command for showing GPU coonfigration 
        result = subprocess.run(command, shell=True, capture_output=True, text=True) # execute a command in the background to show GPU configuration
        
        if result.returncode == 0:
            print(result.stdout) # output after successful execution of command
        else:
            print("- Error message: \n{}".format(result.stderr)) # output after failed exection of command

    else : 

        print("GPU is not available, using CPU instead")