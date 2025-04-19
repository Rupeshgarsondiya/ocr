'''
author: Rupesh Garsondiya
github: @Rupeshgarsondiya
organization: L.J University
'''

import os
import shutil
import pandas as pd

# Load the original CSV file
df = pd.read_csv(os.getcwd()+'/data/raw/train.csv')

# Select 1,000 random samples from the dataset
df_sampled = df.sample(n=750, random_state=42)  # Set random_state for reproducibility

# Save the sampled data to a new CSV file
df_sampled.to_csv(os.getcwd()+"/data/raw/train_sample.csv", index=False)


