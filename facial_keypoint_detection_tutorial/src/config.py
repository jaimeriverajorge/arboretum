# config.py
# Script to define the learning parameters for
# deep learning training and validation
# will also define data paths, and the train
# and validation split ratio

import torch

# constant paths
ROOT_PATH = '../input/facial-keypoints-detection'
OUTPUT_PATH = '../outputs'

# learning parameters

# images are small in dimension (96x96) and are grayscale
# large batch sizes will not cause memory issues
BATCH_SIZE = 256

# learning rate that is the most stable for model and
# dataset, according to tutorial
LR = 0.0001

# will train for 300 epochs
EPOCHS = 300
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train/test split, 80% of data for training, 20% for validation
TEST_SPLIT = 0.2

# show dataset keypoint plot, will see a plot of faces
# along with corresponding keypoints
SHOW_DATASET_PLOT = False
