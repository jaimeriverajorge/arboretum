import torch

# constant paths
ROOT_PATH = '../input'
OUTPUT_PATH = '../outputs'

# learning parameters
BATCH_SIZE = 12
LR = 0.01
EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train/test split
TEST_SPLIT = 0.1

# show dataset keypoint plot
SHOW_DATASET_PLOT = True
