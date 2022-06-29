#!/usr/bin/env python
# coding=utf-8
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
import os
import argparse
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn

from airdet.apis import Trainer
from airdet.config.base import parse_config
from airdet.utils import get_num_devices, synchronize


def make_parser():
    """
    Create a parser with some common arguments used by users.

    Returns:
        argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser("light vison train parser")

    parser.add_argument(
        "-f",
        "--config_file",
        default=None,
        type=str,
        help="plz input your config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main():
    args = make_parser().parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://"
    )
    synchronize()

    config = parse_config(args.config_file)
    config.merge(args.opts)

    trainer = Trainer(config, args)
    trainer.train(args.local_rank)


if __name__ == "__main__":
    main()
