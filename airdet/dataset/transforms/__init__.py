# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
from .transforms import Compose
from .transforms import Resize
from .transforms import RandomHorizontalFlip
from .transforms import ToTensor
from .transforms import Normalize

from .build import build_transforms
