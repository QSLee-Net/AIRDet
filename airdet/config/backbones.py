# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

CSPDarknet = {"name": "CSPDarknet",
            "dep_mul": 0.33,
            "wid_mul": 0.5,
            "out_features": ("dark3", "dark4", "dark5"),
            "depthwise": False,
            "acmix": False,
            "act": "silu",
            }


MobileNet = {"name": "MobileNet",
            "dep_mul": 1,
            "wid_mul": 1,
            "out_features": ("stage2", "stage3", "stage4"),
            }

ShuffleNet = {"name": "ShuffleNet",
            "dep_mul": 1,
            "wid_mul": 1,
            "out_features": ("stage2", "stage3", "stage4"),
            }