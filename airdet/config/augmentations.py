# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

tta = {
        # test time augmentation config
        "nms_thres": 0.65,
        "hflip": True,
        "scales": (448, 512, 576, 640, 704, 768, 832),
        "scales_max_size": 640,
        "scales_hflip": True,
}

SADA = {
        "use_img_lvl": False,
        "use_box_lvl": True,
        "box_prob": 0.3,
        "use_box_color": True,
        "use_box_geo": True,
        "use_dynamic_scale": True,
        "num_subpolicies": 5,
        "scale_splits": [2048, 10240, 51200],
        "autoaug_params": (6, 9, 5, 3,
                           3, 4, 2, 4,
                           4, 4, 5, 2,
                           4, 1, 4, 2,
                           6, 4, 2, 2,
                           2, 6, 2, 2,
                           2, 0, 5, 1,
                           3, 0, 8, 5,
                           2, 8, 7, 5,
                           1, 3, 3, 3),
        }


strong_autoaug = {
         "use_autoaug": True,
         ## mosaic ##
         "mosaic": True,
         "mosaic_prob": 1.0,
         "mosaic_arange": '2x2', # '2x3', '3x3', 'None',
         "mosaic_scale": (0.1, 2),
         "mosaic_size": (640, 640),
         ## mixup ##
         "mixup": True,
         "mixup_prob": 1.0,
         "mixup_scale": (0.5, 1.5),
         ## common transform ##
         "hsv_prob": 1.0,
         "flip_prob": 0.5,
         "degrees": 10.0,
         "translate": 0.1,
         "shear": 2.0,
         "autoaug": SADA,
}


