# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .coco import COCODataset
from .obj365 import Objects365
from .voc import PascalVOCDataset
from .concat_dataset import ConcatDataset
from .mosaic_detection import MosaicDetection
from .custom_voc import CustomVocDataset

__all__ = ["COCODataset", "Objects365", "ConcatDataset", "PascalVOCDataset", "MosaicDetection", "CustomVocDataset"]

class CADVocDataset(CustomVocDataset):
    CLASS2ID = {
        '__bg__': 0,
        'table': 1,
        'bed': 2,
        'pot': 3,
        'basin': 4,
        'flue': 5,
        'aircond': 6,
        'drain': 7,
        'door': 8
    }

