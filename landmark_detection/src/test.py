# ../../../code
# Script to test the output of the trained model
# on different images within the test set
import torch
import numpy as np
import cv2
from torch._C import dtype
import config
import matplotlib.pyplot as plt

from model import FaceKeypointResNet50

model = FaceKeypointResNet50(
    pretrained=False, requires_grad=False).to(config.DEVICE)

# just change the image and model names if wish to use a different one
# most lobes
img_name = 'IA-MG245-W-A'
# least lobes
#img_name = 'IL-SF003-E-A'
# medium lobes
#img_name = 'MO-MG404-N-B'
model_name = 'landmark_output_long_data_with_1s'

# load the model checkpoint
# in this case, using the model filled with 1s, trained for
# 100 epochs
checkpoint = torch.load(
    f'../../../code/{model_name}/model.pth')
# load the model weights and state_dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# load the image into the model

with torch.no_grad():

    image = cv2.imread(f'../input/training/{img_name}.jpg')
    im_h, im_w, im_c = image.shape
    image = cv2.resize(image, (224, 224))
    orig_frame = image.copy()
    orig_h, orig_w, c = orig_frame.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image, dtype=torch.float)
    image = image.unsqueeze(0).to(config.DEVICE)

    outputs = model(image)

outputs = outputs.cpu().detach().numpy()
outputs = outputs.reshape(-1, 2)

keypoints = outputs
orig_frame = cv2.resize(orig_frame, (im_w, im_h))
plt.imshow(orig_frame)
for p in range(keypoints.shape[0]):
    plt.plot(int(keypoints[p, 0]), int(keypoints[p, 1]), 'r.')

plt.savefig(f"{config.OUTPUT_PATH}/test_{img_name}.png")


# cv2.destroyAllWindows()
