import argparse
import tensorflow as tf
import torch
import os
from time import sleep
import numpy as np

"""
class classy:
    def __init__(self):
        self.data = '/Users/josephking/Documents/sponsored_projects/MERGEN/data/vehicle_classifier/scraped_subset'
        self.min_vehicle_area = 400

opt = classy()
"""

def process_images():
    pass


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='path to image directory', required=True)
    parser.add_argument('--min-vehicle-area', type=int, default=400, help='YOLOv5 minimum object size in square pixels, else ignored')

    return parser.parse_args()

def main(opt):

    # Load YOLOv5 weights
    yolov5_weights = torch.hub.load('yolov5', 'custom', path='./yolov5/yolov5s.pt', source='local')

    # Continuously scan image directory for new images
    history = []  # history of all previously seen images
    while True:

        current = os.listdir(opt.data)
        new = [i for i in current if i not in history if "jpg" in i or "png" in i]

        for file in new:

            ##################
            ##### YOLOv5 #####
            ##################

            results = yolov5_weights(os.path.join(opt.data, file))  # run YOLOv5 model

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
            bboxes = coordinates[:, :4]

            # Bounding box area -> keep only those above minimum threshold
            area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])  # Format: xyxy
            bboxes = bboxes[area > opt.min_vehicle_area].astype(int)

            ##################################
            ##### Convert to tf dataset ######
            ##################################

            data = tf.data.Dataset.from_tensor_slices(([np.copy(arr)] * len(bboxes), bboxes))



            history.append(file)

        sleep(1)  # delay 1 second




if __name__ == '__main__':

    opt = parse_opt()
    main(opt)