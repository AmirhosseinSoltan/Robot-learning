import torch
import torch.nn as nn

# Define the CNN module for image processing
class ImageCNN(nn.Module):
    def __init__(self):
        super(ImageCNN, self).__init__()
        # Convolutional layers as shown in the architecture
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        
        # ReLU activations
        self.relu = nn.ReLU()
        
        # Calculate the size of the feature maps after convolutions for 84x84 input
        # After conv1: (84-8)/4+1 = 20
        # After conv2: (20-4)/2+1 = 9
        # After conv3: (9-3)/1+1 = 7
        # So the feature map size is 32 * 7 * 7 = 1568
        self.fc = nn.Linear(32 * 7 * 7, 512)   
    
    def forward(self, x):
        # Applying convolutions with ReLU activations
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # Flattening the output
        x = x.view(x.size(0), -1)
        
        # FC layer
        x = self.relu(self.fc(x))
        
        return x

# Define the state processing module
class StateProcessor(nn.Module):
    def __init__(self):
        super(StateProcessor, self).__init__()
        # Fully connected layers for state processing
        self.fc1 = nn.Linear(8, 64)  # 8 state inputs (x, y, z, theta_x, theta_y, theta_z, theta_w, remaining timesteps)
        self.fc2 = nn.Linear(64, 64)
        
        # Tanh activations
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        # Apply fully connected layers with tanh activations
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        
        return x

# Define the complete ACGD network
class ACGDNetwork(nn.Module):
    def __init__(self):
        super(ACGDNetwork, self).__init__()
        # Image processing module
        self.image_cnn = ImageCNN()
        
        # State processing module
        self.state_processor = StateProcessor()
        
        # Combined processing
        self.fc1 = nn.Linear(512 + 64, 128)  # Concatenated features from image and state
        self.fc2 = nn.Linear(128, 4)  # Output: delta_x, delta_y, delta_z, delta_theta
        
        # Activations
        self.tanh = nn.Tanh()
    
    def forward(self, image, state):
        # Process image
        image_features = self.image_cnn(image)
        
        # Process state
        state_features = self.state_processor(state)
        
        # Concatenate features
        combined = torch.cat((image_features, state_features), dim=1)
        
        # Final processing
        x = self.tanh(self.fc1(combined))
        x = self.fc2(x)  # No activation on the output layer
        
        return x
