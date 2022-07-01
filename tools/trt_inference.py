# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

import argparse
import os

from PIL import Image
from airdet.dataset.transforms import build_transforms
from airdet.utils import postprocess, vis
from airdet.utils import postprocess_gfocal as postprocess_gfocal

from airdet.structures.image_list import to_image_list
from airdet.config.base import parse_config

import torch


COCO_CLASSES = []
for i in range(80):
  COCO_CLASSES.append(str(i))
COCO_CLASSES = tuple(COCO_CLASSES)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def make_parser():
    parser = argparse.ArgumentParser("trt engine inference")

    parser.add_argument("-f", "--config_file", default=None, type=str, help="pls input your config file",)
    parser.add_argument(
        "-t",
        "--trt_path",
        type=str,
        default="lightvision_small.trt",
        help="Input your trt model.",
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
        default=0.6,
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

    # settings
    BATCH_SIZE = 1
    target_dtype = np.float32

    origin_img = np.asarray(Image.open(args.path).convert("RGB"))

    config = parse_config(args.config_file)

    transforms = build_transforms(config, is_train=False)

    img, _ = transforms(origin_img)
    config.dataset.size_divisibility = args.img_size

    img = to_image_list(img, config.dataset.size_divisibility)
    img_np = np.asarray(img.tensors)

    if args.conf is not None:
        config.testing.conf_threshold = args.conf
    if args.nms is not None:
        config.testing.nms_iou_threshold = args.nms

    # set logs
    logger = trt.Logger(trt.Logger.INFO)

    # initialize
    t = open(args.trt_path, 'rb')
    runtime = trt.Runtime(logger)

    model = runtime.deserialize_cuda_engine(t.read())
    context = model.create_execution_context()

    input_batch = img_np.astype(target_dtype)
    output = np.empty([BATCH_SIZE, 8400, config.model.head.num_classes + 5], dtype = target_dtype)

    d_input = cuda.mem_alloc(1 * input_batch.nbytes)
    d_output = cuda.mem_alloc(1 * output.nbytes)

    bindings = [int(d_input), int(d_output)]

    stream = cuda.Stream()

    def predict(batch): # result gets copied into output
        # transfer input data to device
        cuda.memcpy_htod_async(d_input, batch, stream)
        # execute model
        context.execute_async_v2(bindings, stream.handle, None)
        # transfer predictions back
        cuda.memcpy_dtoh_async(output, d_output, stream)
        # syncronize threads
        stream.synchronize()

        return output
        # tensorrt engine inference

    print("Warming up...")
    pred = predict(input_batch)

    # Model Inference
    t0 = time.time()
    pred = predict(input_batch)
    t_all = time.time() - t0
    print("Prediction cost {:.4f}s".format(t_all))

    decode_output = torch.Tensor(pred)
    prediction = postprocess_gfocal(decode_output, config.model.head.num_classes, \
        config.testing.conf_threshold, config.testing.nms_iou_threshold, img)

    ratio = min(origin_img.shape[0] / img.image_sizes[0][0], origin_img.shape[1] / img.image_sizes[0][1])
    bboxes = prediction[0].bbox * ratio
    scores = prediction[0].get_field('scores')
    cls_inds = prediction[0].get_field('labels')

    out_img = vis(origin_img, bboxes, scores, cls_inds, conf=config.testing.conf_threshold, class_names=COCO_CLASSES)

    output_folder = os.path.join(config.miscs.output_dir, "trt_out")
    mkdir(output_folder)
    output_path = os.path.join(output_folder, args.path.split("/")[-1])
    cv2.imwrite(output_path, out_img[:,:,::-1])
