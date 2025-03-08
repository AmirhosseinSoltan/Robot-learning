import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from dataloader import RobotDemonstrationDataset
from network import ACGDNetwork

# Training function
def train_acgd_network(data_dirs, batch_size=32, num_epochs=50, learning_rate=0.0001, save_dir=None, writer=None):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Transforms for the images. Should be (84, 84, 3)
    transform = transforms.Compose([
        transforms.Resize((84, 84)),  # Resize to match the architecture input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create dataset and dataloader
    dataset = RobotDemonstrationDataset(data_dirs, transform=transform)
    
    # Split dataset into train and validation sets (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Initialize the model
    model = ACGDNetwork().to(device)
    
    # Define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for images, states, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            # Move data to device
            images = images.to(device)
            states = states.to(device)
            targets = targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images, states)
            
            # Calculate loss
            loss = loss_fn(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0) # converting the average loss back to the total loss for that batch
        
        epoch_train_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for images, states, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                # Move data to device
                images = images.to(device)
                states = states.to(device)
                targets = targets.to(device)
                
                # Forward pass
                outputs = model(images, states)
                
                # Calculate loss
                loss = loss_fn(outputs, targets)
                
                running_val_loss += loss.item() * images.size(0)
        
        epoch_val_loss = running_val_loss / len(val_dataset)
        val_losses.append(epoch_val_loss)
        
        # Update learning rate
        # scheduler.step(epoch_val_loss)
        
        # Print statistics
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")
        

        # Save model checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
            }, f"{save_dir}/acgd_model_epoch_{epoch+1}.pt")
    

        if writer:
        # Log the final losses to TensorBoard
            for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/validation', val_loss, epoch)  
    writer.close()

    # Save the final model
    torch.save(model.state_dict(), f"{save_dir}/acgd_model_final.pt")
    
    return model, train_losses, val_losses