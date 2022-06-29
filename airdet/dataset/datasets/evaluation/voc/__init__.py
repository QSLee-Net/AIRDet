# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

from loguru import logger
from .voc_eval import do_voc_evaluation


def voc_evaluation(dataset, predictions, output_folder, box_only, **_):
    if box_only:
        logger.warning("voc evaluation doesn't support box_only, ignored.")
    logger.info("performing voc evaluation, ignored iou_types.")
    return do_voc_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
    )
