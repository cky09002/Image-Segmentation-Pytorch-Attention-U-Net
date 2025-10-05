import torch
# Config
LEARNING_RATE = 1e-4  # Increased learning rate
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4  # Smaller batch size for better convergence
NUM_EPOCHS = 20  # More epochs
NUM_WORKERS = 8  # Reduced workers
IMAGE_HEIGHT = 1024  # Higher resolution
IMAGE_WIDTH = 1024   # Square images often work better
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_DIR = "Nature/train/"
TEST_DIR = "Nature/test/"