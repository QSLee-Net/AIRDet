# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import torch
import os
import cv2
import pdb
import numpy as np
from tqdm import tqdm


# electric
cls_max_num = 1
labelidx = {1:'hammer',
            2:'insulation',
            3:'dowel',}
outpath = './showcases/cls3/vis_pred_max%d'%(cls_max_num)
color_dict = {"hammer":(0,255,0), "insulation":(255,0,0), "dowel":(0,0,255)}
os.makedirs(outpath, exist_ok=True)
ref_path = './showcases/cls3/vis_anno'
img_paths = './datasets/VOCelectric/ImageSets/Main/val.txt'
f = open(img_paths)
imglist = f.readlines()
imglist = [os.path.join(ref_path,i.strip()+'.JPG') for i in imglist]
print('img len:',len(imglist))
res_path = './eval_out/electric/predictions.pth'
res = torch.load(res_path)
print('res len:',len(res))


def get_ratio(res_size,img_size):
    img_size = (img_size[1],img_size[0])
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(img_size, res_size))
    ratio = min(ratios)
    return ratio


for p,r in tqdm(zip(imglist,res)):
    img = cv2.imread(p)
    cls = r.get_field('labels').cpu().numpy()
    bbox = r.bbox
    # print(bbox)
    score = r.get_field('scores').cpu().numpy()
    ratio = get_ratio(r.size,img.shape)
    bbox = r.bbox * ratio

    coord_dict={}
    for ss, bb,cc in zip(score,bbox,cls):
        if ss < 0.4: continue
        x1, y1, x2, y2 = map(int,bb)
        category = labelidx[int(cc)]
        if category not in coord_dict.keys():
            coord_dict[category]=[[(x2-x1)*(y2-y1), x1, y1, x2, y2, ss]]
        else:
            coord_dict[category].append([(x2-x1)*(y2-y1), x1, y1, x2, y2, ss])

    for category in coord_dict.keys():
        if category not in os.path.basename(p):
            continue
        color = color_dict[category]
        coord_list = np.array(coord_dict[category])
        coord_list = coord_list[coord_list[:,0].argsort()[::-1]]
        for i in range(cls_max_num):
            if i<len(coord_list):
                x1, y1, x2, y2, ss = coord_list[i,1:]
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                cv2.rectangle(img, pt1=(x1,y1), pt2=(x2,y2), color=color, thickness=4)
                cv2.putText(img, category+"%.3f"%(ss), (x1,y1-3), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, thickness=2)
    cv2.imwrite(os.path.join(outpath,os.path.basename(p)), img)
