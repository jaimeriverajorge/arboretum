# utility functions to help plot the facial keypoints
# on the images of faces

import matplotlib.pyplot as plt
import numpy as np
import config


def valid_keypoints_plot(image, outputs, orig_keypoints, epoch):
    """
    This function plots the regressed (predicted) keypoints
    and the actual keypoints after each validation epoch
    for one image in the batch
    """

    # detach the image, keypoints, and output tensors from GPU
    # to load onto CPU
    image = image.detach().cpu()
    outputs = outputs.detach().cpu().numpy()
    orig_keypoints = orig_keypoints.detach().cpu().numpy()

    # just get a single datapoint from each batch
    img = image[0]
    output_keypoint = outputs[0]
    orig_keypoint = orig_keypoints[0]

    # convert the image to NumPy array format
    img = np.array(img, dtype='float32')
    # transpose image to make channels last
    img = np.transpose(img, (1, 2, 0))
    # reshape to be in original 96x96 dim
    img = img.reshape(96, 96)
    # plot image using Matplotlib
    plt.imshow(img, cmap='gray')

    # reshape predicted and original keypoints to have
    # 2 columns along with respective numbe of rows (hence -1)
    output_keypoint = output_keypoint.reshape(-1, 2)
    orig_keypoint = orig_keypoint.reshape(-1, 2)

    # loop to plot the predicted and original keypoints
    # on the image of the face
    # predicted keypoints are red, actual keypoints are green
    # also plot the corresponding keypiont numbers
    for p in range(output_keypoint.shape[0]):
        plt.plot(output_keypoint[p, 0], output_keypoint[p, 1], 'r.')
        plt.text(output_keypoint[p, 0], output_keypoint[p, 1], f"{p}")
        plt.plot(orig_keypoint[p, 0], orig_keypoint[p, 1], 'g.')
        plt.text(orig_keypoint[p, 0], orig_keypoint[p, 1], f"{p}")

    # save image in outputs folder
    plt.savefig(f"{config.OUTPUT_PATH}/val_epoch_{epoch}.png")
    plt.close()


def test_keypoints_plot(images_list, outputs_list):
    """
    This function plots the keypoints for the outputs
    and images in the 'test.py' script which used the 
    'test.csv' file
    """
    plt.figure(figsize=(10, 10))
    # loop over the images and predicted keypoints
    # in the two lists
    for i in range(len(images_list)):
        outputs = outputs_list[i]
        image = images_list[i]
        outputs = outputs.cpu().detach().numpy()
        outputs = outputs.reshape(-1, 2)
        # use subplot() as we want all the images in
        # a single plot, we will be plotting for
        # 9 images
        plt.subplot(3, 3, i+1)
        plt.imshow(image, cmap='gray')
        for p in range(outputs.shape[0]):
            plt.plot(outputs[p, 0], outputs[p, 1], 'r.')
            plt.text(outputs[p, 0], outputs[p, 1], f"{p}")
        plt.axis('off')
    plt.savefig(f"{config.OUTPUT_PATH}/test_output.png")
    plt.show()
    plt.close()


def dataset_keypoints_plot(data):
    """
    This function shows the image faces and keypoint plots
    that the model will see. This is a good way to validate 
    that our dataset is in fact correct and that the faces
    align with the keypoint features. The plot will be shown
    just before training starts. Press 'q' to quit the plot
    and start training.
    """
    plt.figure(figsize=(20, 40))
    # going to show 30 images
    for i in range(30):
        sample = data[i]
        img = sample['image']
        img = np.array(img, dtype='float32')
        img = np.transpose(img, (1, 2, 0))
        img = img.reshape(96, 96)
        plt.subplot(5, 6, i+1)
        plt.imshow(img, cmap='gray')
        keypoints = sample['keypoints']
        for j in range(len(keypoints)):
            plt.plot(keypoints[j, 0], keypoints[j, 1], 'r.')
    plt.show()
    plt.close()
