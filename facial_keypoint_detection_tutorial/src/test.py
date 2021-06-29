# script to load the test.csv file and predict
# facial keypoints on the unseen images using
# the trained model

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from torch._C import dtype
import utils
import config

from model import FaceKeypointModel
from tqdm import tqdm

# image resize dimension
resize = 96

# initialize the neural network model and load the trained weights
model = FaceKeypointModel().to(config.DEVICE)
# load the model checkpoint
checkpoint = torch.load(f"{config.OUTPUT_PATH}/model.pth")
# load the model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


# load the test.csv file and prepare the image pixels
csv_file = f"{config.ROOT_PATH}/test/test.csv"
data = pd.read_csv(csv_file)
pixel_col = data.Image
image_pixels = []
for i in tqdm(range(len(pixel_col))):
    img = pixel_col[i].split(' ')
    image_pixels.append(img)

# convert to NumPy array
images = np.array(image_pixels, dtype='float32')

# predicting the keypoints for 9 images
images_list, outputs_list = [], []
for i in range(9):
    with torch.no_grad():
        image = images[i]
        image = image.reshape(96, 96, 1)
        image = cv2.resize(image, (resize, resize))
        image = image.reshape(resize, resize, 1)
        orig_image = image.copy()
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float)
        image = image.unsqueeze(0).to(config.DEVICE)

        # forward pass through the model, gets the
        # predicted keypoints and stores in outputs
        outputs = model(image)
        # append the current original image
        images_list.append(orig_image)
        # append the current outputs
        outputs_list.append(outputs)

utils.test_keypoints_plot(images_list, outputs_list)
