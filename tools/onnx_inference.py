#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.


import argparse
import os

import cv2
import numpy as np

import onnxruntime
from PIL import Image
from airdet.dataset.transforms import build_transforms
from airdet.utils import postprocess, vis
from airdet.utils import postprocess_gfocal as postprocess_gfocal
import time

from airdet.structures.image_list import to_image_list

import torch

from airdet.config.base import parse_config


COCO_CLASSES = []
for i in range(80):
  COCO_CLASSES.append(str(i))
COCO_CLASSES = tuple(COCO_CLASSES)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def decode_output(outputs, img_size, p6=False):
    # print(outputs.shape) (1, 2300, 4)

    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs


def make_parser():
    parser = argparse.ArgumentParser("airdet onnx inference")

    parser.add_argument("-f", "--config_file", default=None, type=str, help="pls input your config file",)
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="lightvision_small.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default='./assets/dog.jpg',
        help="Path to your input image.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.05,
        help="confidence threshould to filter the result.",
    )
    parser.add_argument(
        "--nms",
        type=float,
        default=0.7,
        help="nms threshould to filter the result.",
    )
    parser.add_argument(
        "--img_size", 
        type=int, 
        default="640", 
        help="inference image shape"
    )
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()

    input_shape = tuple(args.img_size, args.img_size)
    origin_img = np.asarray(Image.open(args.path).convert("RGB"))

    config = parse_config(args.config_file)

    transforms = build_transforms(config, is_train=False)

    img, _ = transforms(origin_img)
    img = to_image_list(img, config.dataset.size_divisibility)
    img_np = np.asarray(img.tensors)

    if args.conf is not None:
        config.testing.conf_threshold = args.conf
    if args.nms is not None:
        config.testing.nms_iou_threshold = args.nms

    sess_options = onnxruntime.SessionOptions()
    sess_options.intra_op_num_threads = 8
    t0 = time.time()
    session = onnxruntime.InferenceSession(args.model, sess_options=sess_options)
    t_all = time.time() - t0
    print("Prediction cost {:.4f}s".format(t_all))
    ort_inputs = {session.get_inputs()[0].name: img_np}
    output = session.run(None, ort_inputs)

    if config.model.head['name'] == 'GFocalV2':
        decode_output = torch.Tensor(output[0])
        img = to_image_list(img)
        prediction = postprocess_gfocal(decode_output, config.model.head.num_classes, \
            0.3, 0.6, img)
        # BoxList(num_boxes=7, image_width=640, image_height=480, mode=xyxy)
    else:
        decode_output = torch.Tensor(decode_output(output[0], input_shape)) # like yolox
        prediction = postprocess(decode_output, config.model.head.num_classes, \
            config.testing.conf_threshold, config.testing.nms_iou_threshold, img)

    ratio = min(origin_img.shape[0] / img.image_sizes[0][0], origin_img.shape[1] / img.image_sizes[0][1])
    bboxes = prediction[0].bbox * ratio
    scores = prediction[0].get_field('scores')
    cls_inds = prediction[0].get_field('labels')

    out_img = vis(origin_img, bboxes, scores, cls_inds, conf=config.testing.conf_threshold, class_names=COCO_CLASSES)

    output_folder = os.path.join(config.miscs.output_dir, "onnx_out")
    mkdir(output_folder)
    output_path = os.path.join(output_folder, args.path.split("/")[-1])
    cv2.imwrite(output_path, out_img[:,:,::-1])

