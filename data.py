#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[1]:


import json
import random
import os
import cv2
import glob
import math
import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from collections import defaultdict

import torch
from torch.utils.data import Dataset


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True, aug=True):
        self.div = div
        self.aug = aug

    def __call__(self, pic):
        if self.aug:
            img = torch.from_numpy(pic).permute(2,0,1).contiguous()
        else:
            img = torch.from_numpy(pic).permute(2,0,1).contiguous()

        return img.float().div(255) if self.div else img.float()


def jeurissen_conversion(image, segm_contours, conversion, thickness=2):
    assert conversion in ["Color_NoBackground", "WhiteOutline", "BlackOutline", "Color+Background"]

    template_all = np.zeros_like(image) + 255  # initialize the template
    template_all = template_all.astype('int32')

    for contours in segm_contours:
        # Painter's Algorithm (fill first)
        template_all = cv2.drawContours(template_all, [np.array(contour).astype('int32') for contour in contours],
                                        contourIdx=-1, color=(100, 100, 100),
                                        thickness=cv2.FILLED)
        template_all = cv2.drawContours(template_all, [np.array(contour).astype('int32') for contour in contours],
                                        contourIdx=-1, color=(0, 0, 0),
                                        thickness=thickness)

    converted_im = image.copy()
    if conversion == "Color_NoBackground":
        converted_im[template_all == 255] = 255
    elif conversion == "WhiteOutline":
        converted_im[template_all == 0] = 255
        converted_im[template_all != 0] = 0
    elif conversion == "BlackOutline":
        converted_im[template_all == 0] = 0
        converted_im[template_all != 0] = 255

    return converted_im


class CocoDots(Dataset):
    def __init__(self, anns_file, img_dir, size=(150, 150), conversion='WhiteOutline'):
        self.anns_file = anns_file
        self.img_dir = img_dir
        self.size = size
        assert conversion in ["Color_NoBackground", "WhiteOutline", "BlackOutline", "Color+Background"]
        self.conversion = conversion
        self.transform = ToTorchFormatTensor(div=True, aug=False)
        self.dataset = self.read_anns(anns_file)
        self.create_index()

    def read_anns(self, file):
        with open(file, 'rb') as f:
            data = json.load(f)
        return data

    def create_index(self):
        '''
        Adjuted from:
        https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py,
        '''
        # create index
        print('creating index...')
        cats, imgs, img_to_anns = {}, {}, {}
        img_to_serrelab_anns = defaultdict(list)

        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                img_to_anns[ann['image_id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'serrelab_anns' in self.dataset:
            for serrelab_ann in self.dataset["serrelab_anns"]:
                img_to_serrelab_anns[serrelab_ann["image_id"]].append(serrelab_ann)

        print('index created!')

        # create class members
        self.img_to_anns = img_to_anns
        self.img_to_serrelab_anns = img_to_serrelab_anns
        self.imgs = imgs
        self.cats = cats

    def __len__(self):
        return len(self.dataset["serrelab_anns"])

    def __getitem__(self, idx):
        try:
            # Get image and annotations
            serrelab_ann = self.dataset["serrelab_anns"][idx]
            img_data = self.imgs[serrelab_ann["image_id"]]
            ann = self.img_to_anns[serrelab_ann["image_id"]]
            img = Image.open(os.path.join(self.img_dir, img_data["file_name"])).convert('RGB')
            img = np.array(img)

            # Compute resize factors
            h_factor = self.size[1] / img_data["height"]
            w_factor = self.size[0] / img_data["width"]

            # Find resized dot locations
            serrelab_ann_resize = copy.deepcopy(serrelab_ann)
            serrelab_ann_resize["cue_xy"] = [round(serrelab_ann_resize["cue_xy"][0] * w_factor),
                                             round(serrelab_ann_resize["cue_xy"][1] * h_factor)]
            serrelab_ann_resize["fixation_xy"] = [round(serrelab_ann_resize["fixation_xy"][0] * w_factor),
                                                  round(serrelab_ann_resize["fixation_xy"][1] * h_factor)]

            # Convert to an outline image
            things = [segm for segm in ann["segments_info"] if self.cats[segm["category_id"]]["isthing"] == 1]
            segm_contours = [segm["contours_LG"] for segm in things]
            if self.conversion in ['WhiteOutline', 'BlackOutline']:
                thickness = int(1/min(h_factor, w_factor))
            else:
                thickness = 1
            img = jeurissen_conversion(img, segm_contours, self.conversion, thickness=thickness)

            # Resize image
            img = Image.fromarray(img).resize(self.size)
            img = np.array(img)

            # Add dot channel
            dot_channel = np.zeros_like(img[:, :, 0])
            dot_channel[serrelab_ann_resize["fixation_xy"][1], serrelab_ann_resize["fixation_xy"][0]] = 255
            dot_channel[serrelab_ann_resize["cue_xy"][1], serrelab_ann_resize["cue_xy"][0]] = 255
            img = np.concatenate([img, np.expand_dims(dot_channel, 2)], axis=2)

            # Get label
            label = serrelab_ann["same"]

            # Transform
            img = self.transform(img)

            # Gather
            return_dict = {
                "image": img,
                "label": label,
                "id": img_data["id"],
                "index": idx,
                "cue_x": serrelab_ann_resize["cue_xy"][0],
                "cue_y": serrelab_ann_resize["cue_xy"][1],
                "fixation_x": serrelab_ann_resize["fixation_xy"][0],
                "fixation_y": serrelab_ann_resize["fixation_xy"][1]
            }

            return return_dict

        except Exception as e:
            print(
                "Error in getting sample with index {}, image_id {}, and serre_lab sample {}: {}".format(idx, img_data[
                    "id"], serrelab_ann["serrelab_sample"], str(e)))
            print("Sampling new random image")
            new_idx = random.choice(list(range(len(self))))
            return_dict = self.__getitem__(new_idx)
            return return_dict

    def id_to_image(self, image_id):

        serrelab_anns = self.img_to_serrelab_anns[image_id]

        sample_numbers = np.unique([x["serrelab_sample"] for x in serrelab_anns]).tolist()
        img_data = self.imgs[image_id]
        I = Image.open(os.path.join(self.img_dir, img_data["file_name"])).convert('RGB')
        I = np.array(I)
        h, w, c = I.shape
        aspect_ratio = h / w

        fig, ax = plt.subplots(nrows=len(sample_numbers), ncols=2, gridspec_kw={'wspace': 0.00005, 'hspace': 0.05},
                               figsize=(15, aspect_ratio * 15 * 2))

        for i in range(len(sample_numbers)):
            sample_number = sample_numbers[i]
            positive = [x for x in serrelab_anns if x["serrelab_sample"] == sample_number and x["same"] == 1][0]
            negative = [x for x in serrelab_anns if x["serrelab_sample"] == sample_number and x["same"] == 0][0]
            pos_I = I.copy()
            neg_I = I.copy()

            cv2.circle(pos_I, positive["fixation_xy"], 5, (0, 0, 255), cv2.FILLED, 8, 0)
            cv2.circle(neg_I, negative["fixation_xy"], 5, (0, 0, 255), cv2.FILLED, 8, 0)

            cv2.circle(pos_I, positive["cue_xy"], 5, (0, 255, 0), cv2.FILLED, 8, 0)
            cv2.circle(neg_I, negative["cue_xy"], 5, (255, 0, 0), cv2.FILLED, 8, 0)

            ax[i, 0].imshow(pos_I)
            ax[i, 0].axis('off')
            ax[i, 1].imshow(neg_I)
            ax[i, 1].axis('off')

    def tensor_to_image(self, img_w_dots):
        if img_w_dots.is_cuda:
            img_cpu = img_w_dots.detach().cpu().numpy()
        else:
            img_cpu = img_w_dots.detach().numpy()
        img = np.transpose(img_cpu[0:3, :, :], (1, 2, 0)) * 255
        img = img.astype('int32')
        img = img.copy()
        dots = img_cpu[3, :, :] * 255
        y_values, x_values = np.where(dots != 0)
        dot_positions = list(zip(x_values.tolist(), y_values.tolist()))
        for dot_position in dot_positions:
            img = cv2.circle(img, dot_position, radius=3, color=(255, 0, 0), thickness=cv2.FILLED)
        return img




