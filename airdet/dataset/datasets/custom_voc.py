#coding=utf-8
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
import os

import torch
import torch.utils.data
import cv2

from loguru import logger

import sys
import numpy as np
from collections import defaultdict
from PIL import Image

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from airdet.structures.bounding_box import BoxList
from airdet.structures.boxlist_ops import cat_boxlist


class CustomVocDataset(torch.utils.data.Dataset):
    #NOTE: several class names may correspond to same class_id, e.g., 'person' and 'body'
    CLASS2ID = {
        "__background__ ": 0,
    }

    def __init__(self, data_dir, split, CLASS2ID, MIN_BBOX_SIZE=2, ignored_cls_names=[], use_difficult=False, transforms=None,
        b_filter_empty=True, MIN_BBOX_AREA=1,
        **kwargs,
    ):
        self.root = data_dir
        self.image_set = split
        self.keep_difficult = use_difficult
        self._transforms = transforms
        self.b_filter_empty = b_filter_empty

        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")

        self.class_to_ind = CLASS2ID
        self.ind_to_class = {v: k for k, v in self.class_to_ind.items()}

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]

        self.ignored_cls_names = ignored_cls_names
        self.MIN_BBOX_SIZE = MIN_BBOX_SIZE # negative for not filtering
        self.MIN_BBOX_AREA = MIN_BBOX_AREA # negative for not filtering

        # remove empty annotations
        ids = []
        cls_box_num = defaultdict(int)
        for index, _id in enumerate(self.ids):
            target = self.get_groundtruth(index)
            target = target.clip_to_image(remove_empty=True)

            if self.b_filter_empty and len(target.bbox) <= 0:
                continue
            ids.append(_id)
            if len(target.bbox) > 0:
                labels = target.get_field('labels')
                for l in labels:
                    cls_box_num[self.ind_to_class[int(l)]] += 1

        logger.info('*'*80)
        logger.info('total images | valid images: {} | {}'.format(len(self.ids), len(ids)))
        logger.info('  b_filter_empty: {}'.format(self.b_filter_empty))
        for cls_name, num in cls_box_num.items():
            logger.info('  {}: {}'.format(cls_name, num))
        logger.info('  ignored_cls_names: {}'.format(self.ignored_cls_names))
        logger.info('  min_bbox_area: {}'.format(self.MIN_BBOX_SIZE))
        logger.info('  min_bbox_ratio: {}'.format(self.MIN_BBOX_AREA))
        logger.info('')
        self.cls_box_num = cls_box_num

        self.ids = ids
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

    def __getitem__(self, inp):
        if type(inp) is tuple:
            index = inp[1]
        else:
            index = inp

        img_id = self.ids[index]
        _img_file = self._imgpath % img_id
        if not os.path.isfile(_img_file):
            for ext in ['.png', '.jpeg', '.JPG']:
                if os.path.isfile(_img_file.replace('.jpg', ext)):
                    _img_file = _img_file.replace('.jpg', ext)
                    break

        img = Image.open(_img_file).convert("RGB")
        target = self.get_groundtruth(index)
        target = target.clip_to_image(remove_empty=True)

        # PIL to numpy array
        img = np.asarray(img)      # rgb

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, index

    def pull_item(self, index):
        img_id = self.ids[index]

        _img_file = self._imgpath % img_id
        if not os.path.isfile(_img_file):
            for ext in ['.png', '.jpeg', '.JPG']:
                if os.path.isfile(_img_file.replace('.jpg', ext)):
                    _img_file = _img_file.replace('.jpg', ext)
                    break

        img = Image.open(_img_file).convert("RGB")
        target = self.get_groundtruth(index)
        target = target.clip_to_image(remove_empty=True)

        classes = target.get_field("labels")
        res = np.zeros((len(target.bbox), 5))
        for idx in range(len(target.bbox)):
            res[idx, 0:4] = target.bbox[idx]
            res[idx, 4] = classes[idx]

        # PIL to numpy array
        img = np.asarray(img)      # rgb

        return img, res, index


    def __len__(self):
        return len(self.ids)

    def get_groundtruth(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(anno)

        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        target.add_field("difficult", anno["difficult"])
        return target

    def load_anno(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(anno)

        return anno["labels"]

    def _map_cls_name(self, cls_name):
        return cls_name

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 0

        size = target.find("size")
        im_info = tuple(map(int, (float(size.find("height").text), float(size.find("width").text))))

        for obj in target.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.lower().strip()
            name = self._map_cls_name(name)

            if name not in self.class_to_ind.keys():
                if name not in self.ignored_cls_names:
                    self.ignored_cls_names.append(name)
                continue
            bb = obj.find("bndbox")
            box = [
                bb.find("xmin").text,
                bb.find("ymin").text,
                bb.find("xmax").text,
                bb.find("ymax").text,
            ]

            box = map(float, box)
            bndbox = tuple(
                map(lambda x: x - TO_REMOVE, list(map(int, box)))
            )

            box_size = (bndbox[2] - bndbox[0] + 1, bndbox[3] - bndbox[1] + 1)
            if self.MIN_BBOX_AREA==1:
                # for filter small objects
                if box_size[0]<self.MIN_BBOX_SIZE or box_size[1] < self.MIN_BBOX_SIZE:
                    continue
            else:
                # filter for small objects with the area requirement.
                if box_size[0]*box_size[1]<(float(size.find("height").text)*float(size.find("width").text)/self.MIN_BBOX_AREA):
                    continue
            #
            boxes.append(bndbox)
            gt_classes.append(self.class_to_ind[name])
            difficult_boxes.append(difficult)

        if len(boxes) == 0:
            #print('Warning: {} has no annotations'.format(target.find('filename').text))
            boxes = np.zeros((0, 4))

        gt_labels = torch.tensor(gt_classes)
        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "difficult": torch.tensor(difficult_boxes),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        img_id = self.ids[index]
        try:
            anno = ET.parse(self._annopath % img_id).getroot()
            size = anno.find("size")
            im_info = tuple(map(int, (float(size.find("height").text), float(size.find("width").text))))
        except:
            logger.info('{} parse xml error !!'.format(img_id))
            _img_file = self._imgpath % img_id
            if not os.path.isfile(_img_file):
                for ext in ['.png', '.jpeg', '.JPG']:
                    if os.path.isfile(_img_file.replace('.jpg', ext)):
                        _img_file = _img_file.replace('.jpg', ext)
                        break
            img = cv2.imread(_img_file)
            _height, _width = img.shape[:2]
            im_info = (_height, _width)

        return {"height": im_info[0], "width": im_info[1]}

    def map_class_id_to_class_name(self, class_id):
        return self.ind_to_class[class_id]
