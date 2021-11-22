import argparse
import tensorflow as tf
import torch
import os
from time import sleep
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

"""
class classy:
    def __init__(self):
        self.input_dir = '/Users/josephking/Documents/sponsored_projects/MERGEN/data/vehicle_classifier/scraped_subset'
        self.min_vehicle_area = 400
        self.batch_size = 64
        self.output_dir = '/Users/josephking/Documents/sponsored_projects/MERGEN/data/vehicle_classifier/scraped_output'

opt = classy()
"""


def bbox_crop(image: tf.Tensor, offset_height: int, offset_width: int, target_height: int, target_width: int):
    """
    Crops an image according to bounding box coordinates
    :param image: 4-D Tensor of shape [batch, height, width, channels] or 3-D Tensor of shape [height, width, channels]
    :param offset_height: Vertical coordinate of the top-left corner of the bounding box in image.
    :param offset_width: Horizontal coordinate of the top-left corner of the bounding box in image.
    :param target_height: Height of the bounding box.
    :param target_width: Width of the bounding box.
    :return: 4- or 3-D tensor
    """
    return tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)

def resize(image: tf.Tensor, height: int, width: int):
    """
    :param image: 4d tensor of shape [batch, height, width, channels] or 3-D Tensor of shape [height, width, channels]
    :param height: int, pixel height
    :param width: int, pixel width
    :return: A tensor of desired resized dimensions
    """
    return tf.image.resize(image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

def process_image(image: tf.Tensor, bboxes: tf.Tensor):
    """
    Processes tensor of images
    :param image: 4-d tensor of shape [batch, heigh, width, channels]
    :param bboxes: 2-d tensor of shape [batch, n_vehicles_per_image]
    :return: 4-d tensor of image tensors
    """
    image = bbox_crop(image, bboxes[0], bboxes[1], bboxes[2], bboxes[3])
    image = resize(image, height=224, width=224)  # imagenet uses 224 x 224 so we stick with this
    return image

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, help='directory path where input images located', required=True)
    parser.add_argument('--min-vehicle-area', type=int, default=400, help='YOLOv5 minimum object size in square pixels, else ignored')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size for vehicle make-model classification')
    parser.add_argument('--output-dir', type=str, help='directory path to where output images should be placed')
    return parser.parse_args()

def main(opt):

    # Load YOLOv5 weights
    yolov5_weights = torch.hub.load('yolov5', 'custom', path='./yolov5/yolov5s.pt', source='local')

    # Load Make-Model-Classifier weights
    mm_weights = tf.keras.models.load_model('model_weights')

    # Load label mapping
    with open('label_mapping.json') as f:
        label_map = json.load(f)

    # Continuously scan image directory for new images
    history = []  # history of all previously seen images
    while True:

        current = os.listdir(opt.input_dir)
        new = [i for i in current if i not in history if "jpg" in i or "png" in i]

        for file in new[:9]:

            ##################
            ##### YOLOv5 #####
            ##################

            results = yolov5_weights(os.path.join(opt.input_dir, file))  # run YOLOv5 model

            arr = np.squeeze(np.array(results.imgs))  # return 3-d np.array of image
            coordinates = results.xyxy[0].numpy()  # bounding box coordinates for all objects

            # Check if any objects detected, else move on to next image
            if len(coordinates) == 0:
                continue

            # Keep only: 2 (cars), 5 (bus), 7 (truck)
            coordinates = coordinates[(coordinates[:, -1] == 2.0) | (coordinates[:, -1] == 5.0) | (coordinates[:, -1] == 7.0)]

            # If no cars/bus/truck, move on to next image
            if len(coordinates) == 0:
                continue

            # Restict to bounding boxes themselves
            bboxes = coordinates[:, :4]  # Format: xyxy

            # Bounding box area -> keep only those above minimum threshold
            area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
            bboxes = bboxes[area > opt.min_vehicle_area]

            # If none adequately size, move on to next image
            if len(bboxes) == 0:
                continue

            # Cast to int
            bboxes = bboxes.astype(int)

            # Rearrange bounding boxes to tensorflow's preferred: y1, x1, y2-y1, x2-x1
            yx = np.concatenate((bboxes[:, 1].reshape(len(bboxes), 1), bboxes[:, 0].reshape(len(bboxes), 1)), axis=1)
            y_delta = np.subtract(bboxes[:, 3], bboxes[:, 1]).reshape(len(bboxes), 1)
            x_delta = np.subtract(bboxes[:, 2], bboxes[:, 0]).reshape(len(bboxes), 1)
            bboxes_rearranged = np.concatenate((yx, y_delta, x_delta), axis=1)

            ##################################
            ##### Convert to tf dataset ######
            ##################################

            arr_4d = np.array([np.copy(arr)] * len(bboxes))  # batched in case > 1 vehicle in image
            data = tf.data.Dataset.from_tensor_slices((arr_4d, tf.cast(bboxes_rearranged, tf.int32)))
            data = data.map(process_image, num_parallel_calls=tf.data.AUTOTUNE).batch(opt.batch_size)

            # Make prediction
            pred = mm_weights.predict(data)

            # Get argmax per object in image
            argmax0 = np.argmax(pred, axis=1)

            # Affix labels
            labels = [label_map[str(i)] for i in argmax0]

            # Affix bounding boxes to image
            plt.cla()
            ax = plt.gca()
            ax.imshow(arr)
            for rect in bboxes_rearranged:
                rec = Rectangle((rect[1], rect[0]), rect[3], rect[2], linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rec)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(opt.output_dir, file), bbox_inches='tight', pad_inches=0)

            history.append(file)

        sleep(1)  # delay 1 second before checking if more images




if __name__ == '__main__':

    opt = parse_opt()
    main(opt)