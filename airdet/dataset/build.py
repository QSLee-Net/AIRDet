# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import bisect
import copy
from loguru import logger
import math

import torch.utils.data
from airdet.utils import get_world_size
from airdet.utils import import_file

from . import datasets as D
from .datasets import MosaicDetection
from . import samplers

from .collate_batch import BatchCollator, TTACollator
from .transforms import build_transforms


def build_dataset(cfg, dataset_list, transforms, dataset_catalog, is_train=True, training_mosaic=False):
    """
    Arguments:
        dataset_list (list[str]): Contains the names of the datasets, i.e.,
            coco_2014_trian, coco_2014_val, etc
        transforms (callable): transforms to apply to each (image, target) sample
        dataset_catalog (DatasetCatalog): contains the information on how to
            construct a dataset.
        is_train (bool): whether to setup the dataset for training or testing
    """
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    datasets = []
    for dataset_name in dataset_list:
        # read data from config first
        data = cfg.get_data(dataset_name)
        if data is None:
            # read data from data_catalog
            data = dataset_catalog.get(dataset_name)
        if data is None:
            raise RuntimeError("Dataset not available: {}".format(dataset_name))

        factory = getattr(D, data["factory"])
        args = data["args"]
        # for COCODataset, we want to remove images without annotations
        # during training
        if data["factory"] == "COCODataset":
            args["remove_images_without_annotations"] = False # is_train
        if data["factory"] == "PascalVOCDataset":
            args["use_difficult"] = not is_train
        args["transforms"] = transforms
        # make dataset from factory
        dataset = factory(**args)

        # mosaic wrapped
        if is_train and training_mosaic:
            dataset = MosaicDetection(dataset=dataset, img_size=cfg.training.augmentation.mosaic_size, transforms=transforms, enable_mixup=cfg.training.augmentation.mixup)

        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)

    return [dataset]


def make_data_sampler(dataset, shuffle):

    return samplers.DistributedSampler(dataset, shuffle=shuffle)


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(
    dataset, sampler, aspect_grouping, images_per_batch, num_iters=None, start_iter=0, training_mosaic = False
):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, images_per_batch, drop_uneven=False
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=False
        )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter, enable_mosaic = training_mosaic
        )
    return batch_sampler


def make_data_loader(cfg, is_train=True):
    training_mosaic = cfg.training.augmentation.mosaic
    num_gpus = get_world_size()
    if is_train:
        images_per_batch = cfg.training.images_per_batch
        assert (
            images_per_batch % num_gpus == 0
        ), "training_imgs_per_batch ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = True
    else:
        images_per_batch = cfg.testing.images_per_batch
        assert (
            images_per_batch % num_gpus == 0
        ), "testing_imgs_per_batch ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False
        start_iter = 0

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1] if cfg.dataset.aspect_ratio_grouping else []

    paths_catalog = import_file(
        "airdet.config.paths_catalog", cfg.dataset.paths_catalog, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset_list = cfg.dataset.train_ann if is_train else cfg.dataset.val_ann

    datasets = build_dataset(cfg, dataset_list, None, DatasetCatalog, is_train, training_mosaic=training_mosaic)

    # calculate total iters
    if is_train:
        cfg.training.iters_per_epoch = math.ceil(len(datasets[0]) / images_per_batch)
        num_iters = cfg.training.total_epochs * cfg.training.iters_per_epoch
        start_iter = cfg.training.start_epoch * cfg.training.iters_per_epoch
    else:
        num_iters = None
        start_iter = 0

    transforms = None if not is_train and cfg.testing.use_tta else build_transforms(cfg, is_train)
    for dataset in datasets:
        dataset._transforms = transforms
        if hasattr(dataset, '_dataset'):
            dataset._dataset._transforms = transforms

    data_loaders = []
    for dataset in datasets:
        sampler = make_data_sampler(dataset, shuffle)
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter, training_mosaic
        )
        collator = TTACollator() if not is_train and cfg.testing.use_tta else \
            BatchCollator(cfg.dataset.size_divisibility)
        num_workers = cfg.dataset.num_workers
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )
        data_loaders.append(data_loader)
    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders
