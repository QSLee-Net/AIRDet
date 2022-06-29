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
def trt_speed(trt_path, h, w):

    # settings
    target_dtype = np.float32

    # set logs
    Logger = trt.Logger(trt.Logger.INFO)

    # initialize
    t = open(trt_path, 'rb')
    runtime = trt.Runtime(Logger)

    model = runtime.deserialize_cuda_engine(t.read())
    context = model.create_execution_context()

    input_batch = np.ones([1, 3, h, w], dtype = target_dtype)
    output = np.empty([1, 8400, 85], dtype = target_dtype)

    d_input = cuda.mem_alloc(1 * input_batch.nbytes)
    d_output = cuda.mem_alloc(1 * output.nbytes)

    bindings = [int(d_input), int(d_output)]

    stream = cuda.Stream()

    def predict(batch): # result gets copied into output
        # transfer input data to device
        cuda.memcpy_htod_async(d_input, batch, stream)
        # execute model
        context.execute_async_v2(bindings, stream.handle, None)  # 此处采用异步推理。如果想要同步推理，需将execute_async_v2替换成execute_v2
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
    logger.info("Model inference time {:.4f}s / img per device".format(t_all / 500))
