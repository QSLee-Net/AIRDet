# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import os

import torch
from tqdm import tqdm
from loguru import logger

from airdet.dataset.datasets.evaluation import evaluate
from airdet.utils import (
    is_main_process,
    get_world_size,
    all_gather,
    synchronize
 )
from airdet.utils.timer import Timer, get_time_str
from airdet.dataset.transforms.tta_aug import im_detect_bbox_aug

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import numpy as np
from airdet.utils import postprocess_gfocal as postprocess_gfocal
from airdet.structures.image_list import to_image_list

def compute_on_dataset(config, context, data_loader, device, timer=None, tta=False):

    COCO_CLASSES = []
    for i in range(80):
        COCO_CLASSES.append(str(i))
    COCO_CLASSES = tuple(COCO_CLASSES)

    results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            images = to_image_list(images)
            img = images
            img_np = images.tensors.numpy()

            input_batch = img_np.astype(np.float32)
            trt_output = np.empty([config.testing.images_per_batch, 8400, config.model.head.num_classes + 5], dtype = np.float32)

            d_input = cuda.mem_alloc(1 * input_batch.nbytes)
            d_output = cuda.mem_alloc(1 * trt_output.nbytes)
            bindings = [int(d_input), int(d_output)]
            stream = cuda.Stream()

            def predict(batch): # result gets copied into output
                # transfer input data to device
                cuda.memcpy_htod_async(d_input, batch, stream)
                # execute model
                context.execute_async_v2(bindings, stream.handle, None)
                # transfer predictions back
                cuda.memcpy_dtoh_async(trt_output, d_output, stream)
                # syncronize threads
                stream.synchronize()
                return trt_output

            pred = predict(input_batch)
            decode_output = torch.Tensor(pred)
            output = postprocess_gfocal(decode_output, config.model.head.num_classes, \
                config.testing.conf_threshold, config.testing.nms_iou_threshold, img)
            if timer:
                torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) if o is not None else o for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu, multi_gpu_infer):
    if multi_gpu_infer:
        all_predictions = all_gather(predictions_per_gpu)
    else:
        all_predictions = [predictions_per_gpu]
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        config,
        context,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        multi_gpu_infer=True,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(config, context, data_loader, device, inference_timer)
    # wait for all processes to complete before measuring the time
    if multi_gpu_infer:
        synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)

    total_infer_time = get_time_str(inference_timer.total_time)

    predictions = _accumulate_predictions_from_multiple_gpus(predictions, multi_gpu_infer)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
