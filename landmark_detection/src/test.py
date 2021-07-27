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

resize = 224

# just change the image and model names if wish to use a different one
# most lobes
img_name = 'IA-MG245-W-A'
# least lobes
#img_name = 'IL-SF003-E-A'
# medium lobes
#img_name = 'MO-MG404-N-B'
model_name = 'lobe_landmark_output_filled_with_1s_1000_epochs_lr001'

# true test image
#img_name = 'IA-MG244-S-C'

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
    # for getting an image that was used for training
    image = cv2.imread(f'../input/training/{img_name}.jpg')

    # for getting an image that is brand new (true test)
    # coming from the small res images, since the data
    # filled with 1s was trained on the small res
    # image = cv2.imread(
    #    f'../../../data-extraction/oak_images_small/{img_name}.jpg')

    im_h, im_w, im_c = image.shape
    image = cv2.resize(image, (resize, resize))
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
keypoints = keypoints * [im_w / resize, im_h / resize]
orig_frame = cv2.resize(orig_frame, (im_w, im_h))
plt.imshow(orig_frame)
for p in range(keypoints.shape[0]):
    plt.plot(int(keypoints[p, 0]), int(keypoints[p, 1]), 'r.')
    # cv2.circle(orig_frame, (int(keypoints[p, 0]), int(
    #    keypoints[p, 1])), 1, (0, 0, 255), -1, cv2.LINE_AA)

#orig_frame = cv2.resize(orig_frame, (im_w, im_h))
#cv2.imshow('landmark points:', orig_frame)
# cv2.waitKey(0)
plt.savefig(f"{config.OUTPUT_PATH}/test_{img_name}.png")
plt.show()

# cv2.destroyAllWindows()
