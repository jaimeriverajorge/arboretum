# python file for keypoint and bounding box detection in images
import torch
import numpy as np
import cv2
import argparse
import utils

from PIL import Image
from torchvision.transforms import transforms as transforms
from models import get_model

# construct the argument parser to parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True,
                    help='path to the input data')
args = vars(parser.parse_args())

# transform to convert the image to tensor
transform = transforms.Compose([transforms.ToTensor()])

# set the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the model on to the computation device and set to eval mode
model = get_model().to(device).eval()

image_path = args['input']
image = Image.open(image_path).convert('RGB')
# numpy copy of the image for OpenCV functions
orig_numpy = np.array(image, dtype=np.float32)
# convert the numpy image to OpenCV BGR format
orig_numpy = cv2.cvtColor(orig_numpy, cv2.COLOR_RGB2BGR) / 255.
# transform the image
image = transform(image)

# add a batch dimension
image = image.unsqueeze(0).to(device)

# get the detections, forward pass through the model
with torch.no_grad():
    outputs = model(image)

# draw the keypoints, lines, and bounding boxes
output_image = utils.draw_keypoints_and_boxes(outputs, orig_numpy)

# final step: show the output on the screen and save it to the disk
# visualize the image
cv2.imshow('Keypoint image', output_image)
cv2.waitKey(0)

# set the save path
save_path = f"../outputs/{args['input'].split('/')[-1].split('.')[0]}.jpg"
cv2.imwrite(save_path, output_image*255.)
