#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=8 --master_port=29501 tools/train.py -f configs/airdet_m.py
