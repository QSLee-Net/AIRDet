# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

import argparse
import os

import torch
from loguru import logger

@logger.catch
def trt_speed(trt_path, batch_size, h, w):

    # settings
    target_dtype = np.float32

    # set logs
    Logger = trt.Logger(trt.Logger.INFO)

    # initialize
    t = open(trt_path, 'rb')
    runtime = trt.Runtime(Logger)

    model = runtime.deserialize_cuda_engine(t.read())
    context = model.create_execution_context()

    input_batch = np.ones([batch_size, 3, h, w], dtype = target_dtype)
    output = np.empty([batch_size, 8400, config.model.head.num_classes + 5], dtype = target_dtype)

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

    # Warming up
    pred = predict(input_batch)

    # Model Inference
    t0 = time.time()
    for i in range(500):
        pred = predict(input_batch)
    t_all = time.time() - t0
    logger.info("Model inference time {:.4f}s / img per device".format(t_all / 500 / batch_size))
