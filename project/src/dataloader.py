
import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


# Define the dataset class
class RobotDemonstrationDataset(Dataset):
    def __init__(self, data_dirs, transform=None):
        """
        Args:
            data_dirs (list): List of directories containing the data
            transform (callable, optional): Optional transform to be applied on images
        """
        self.transform = transform
        self.samples = []
        
        # Process each data directory
        for data_dir in data_dirs:
            csv_file = os.path.join(data_dir, 'end_effector_poses.csv')
                        
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Calculate delta values (target) from consecutive poses
            # We'll use the current state and image to predict the delta to the next state
            for i in range(len(df) - 1):
                current_row = df.iloc[i]
                next_row = df.iloc[i+1]
                
                # Get image path
                image_name = current_row['Image_Number']
                image_path = os.path.join(data_dir, f"{image_name}.png")
                
                if not os.path.exists(image_path):
                    continue
                
                # Current state 
                current_state = [
                    current_row['x'], ## gripper x position
                    current_row['y'], 
                    current_row['z'],
                    current_row['theta_x'],## gripper x rotation
                    current_row['theta_y'],
                    current_row['theta_z'],
                    current_row['theta_w'], 
                    (len(df) - i) / len(df) ## the remaining timesteps of the episode normalized to the range [0;1].
                ]
                
                # Calculate delta values (target)
                delta_x = next_row['x'] - current_row['x']
                delta_y = next_row['y'] - current_row['y']
                delta_z = next_row['z'] - current_row['z']
                delta_theta = next_row['theta_w'] - current_row['theta_w']  
                
                # Target values
                target = [delta_x, delta_y, delta_z, delta_theta]
                
                # Add to samples
                self.samples.append({
                    'image_path': image_path,
                    'state': current_state,
                    'target': target
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Convert state and target to tensors
        state = torch.tensor(sample['state'], dtype=torch.float32)
        target = torch.tensor(sample['target'], dtype=torch.float32)
        
        return image, state, target