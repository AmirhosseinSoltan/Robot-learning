import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from network import ACGDNetwork
import pandas as pd
import random

def visualize_prediction(image, state, true_delta, pred_delta):
    """
    Visualize the model's prediction compared to the ground truth.
    
    Args:
        image: The input image
        state: The input state
        true_delta: The ground truth delta values
        pred_delta: The predicted delta values
    """
    # Convert image tensor to numpy for visualization
    if isinstance(image, torch.Tensor):
        # Denormalize the image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image * std + mean
        image = image.permute(1, 2, 0).cpu().numpy()
        image = np.clip(image, 0, 1)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(12, 8))
    
    # Plot the image
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(image)
    ax1.set_title("Input Image")
    ax1.axis('off')
    
    # Plot the state values
    ax2 = fig.add_subplot(2, 2, 2)
    state_labels = ['x', 'y', 'z', 'θx', 'θy', 'θz', 'θw','TimeSteps']
    state_values = state.cpu().numpy() if isinstance(state, torch.Tensor) else state
    ax2.bar(state_labels, state_values)
    ax2.set_title("Input State")
    
    # Plot the delta values (true vs predicted)
    ax3 = fig.add_subplot(2, 2, 3)
    delta_labels = ['Δx', 'Δy', 'Δz', 'Δθ']
    
    true_values = true_delta.cpu().numpy() if isinstance(true_delta, torch.Tensor) else true_delta
    pred_values = pred_delta.cpu().numpy() if isinstance(pred_delta, torch.Tensor) else pred_delta
    
    x = np.arange(len(delta_labels))
    width = 0.35
    
    ax3.bar(x - width/2, true_values, width, label='Ground Truth')
    ax3.bar(x + width/2, pred_values, width, label='Prediction')
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(delta_labels)
    ax3.set_title("Delta Values Comparison")
    ax3.legend()
    
    # Plot the error
    ax4 = fig.add_subplot(2, 2, 4)
    error = np.abs(true_values - pred_values)
    ax4.bar(delta_labels, error)
    ax4.set_title("Absolute Error")
    
    plt.tight_layout()
    plt.savefig('prediction_visualization.png')
    plt.show()

def test_model(model_path, data_dirs, num_samples=5):
    """
    Test the trained model on random samples from the dataset.
    
    Args:
        model_path: Path to the trained model
        data_dirs: List of data directories
        num_samples: Number of samples to test
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the model
    model = ACGDNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Define transforms for the images
    transform = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Collect test samples
    test_samples = []
    
    for data_dir in data_dirs:
        csv_file = os.path.join(data_dir, 'end_effector_poses.csv')
        if not os.path.exists(csv_file):
            continue
            
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Calculate delta values (target) from consecutive poses
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
                current_row['x'], 
                current_row['y'], 
                current_row['z'],
                current_row['theta_x'],
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
            
            # Add to test samples
            test_samples.append({
                'image_path': image_path,
                'state': current_state,
                'target': target
            })
    
    # Randomly select samples for testing
    if len(test_samples) > num_samples:
        test_samples = random.sample(test_samples, num_samples)
    
    # Test the model on each sample
    mse_losses = []
    
    for i, sample in enumerate(test_samples):
        # Load and preprocess image
        image = Image.open(sample['image_path']).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Convert state to tensor
        state_tensor = torch.tensor(sample['state'], dtype=torch.float32).unsqueeze(0).to(device)
        
        # Convert target to tensor
        target_tensor = torch.tensor(sample['target'], dtype=torch.float32)
        
        # Get model prediction
        with torch.no_grad():
            pred_tensor = model(image_tensor, state_tensor).squeeze(0).cpu()
        
        # Calculate MSE loss
        mse_loss = torch.nn.functional.mse_loss(pred_tensor, target_tensor).item()
        mse_losses.append(mse_loss)
        
        # Visualize the prediction
        print(f"\nSample {i+1}:")
        print(f"Image: {sample['image_path']}")
        print(f"State: {sample['state']}")
        print(f"True Delta: {sample['target']}")
        print(f"Predicted Delta: {pred_tensor.tolist()}")
        print(f"MSE Loss: {mse_loss:.6f}")
        
        # Visualize the first few samples
        if i < 5:
            visualize_prediction(
                image_tensor.squeeze(0),
                state_tensor.squeeze(0),
                target_tensor,
                pred_tensor
            )
    
    # Print average MSE loss
    avg_mse = sum(mse_losses) / len(mse_losses)
    print(f"\nAverage MSE Loss: {avg_mse:.6f}")

if __name__ == "__main__":
    # Define the data directories
    data_dirs = [
        "../dataset/task_1_bag_2_data",
        "../dataset/task_1_bag_3_data",
        "../dataset/task_1_bag_4_data"
    ]
    
    # Path to the trained model
    model_path = "../checkpoints/localization.pt"
    
    # Test the model
    test_model(model_path, data_dirs, num_samples=5) 