import os
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from training_engine import train_acgd_network


if __name__ == "__main__":
    # Data directories
    data_dirs = [
            "../dataset/task_1_bag_2_data",
            "../dataset/task_1_bag_3_data",
            "../dataset/task_1_bag_4_data"
        ]
    
    #Creating directories for checkpoints and log files
    root = Path("../")
    Experiment = 'experiment1'  # TODO: CHANGE THIS every time before running a new training!
    save_dir = root/'checkpoints'/Experiment
    if not os.path.exists(save_dir):
        os.makedirs(save_dir,exist_ok=True)
    log_dir = os.path.join(root/'log_dir', Experiment)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir,exist_ok=True)

    # Initialize TensorBoard
    writer = SummaryWriter(log_dir=log_dir)  

    # Training the model...
    model, train_losses, val_losses = train_acgd_network(
        data_dirs=data_dirs,
        batch_size=32,
        num_epochs=250,
        learning_rate=0.001,
        save_dir=save_dir,
        writer=writer
    )
    
    print("Training completed!") 