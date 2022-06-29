# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import torch
import os
import cv2
import pdb
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET

# electric
cls_max_num = 2
labelidx = {1:'hammer',
            2:'insulation',
            3:'dowel',}
outpath = './showcases/cls3/vis_pred_max%d'%(cls_max_num)
color_dict = {"hammer":(0,255,0), "insulation":(255,0,0), "dowel":(0,0,255)}

os.makedirs(outpath, exist_ok=True)
root_path = "./datasets/VOCelectric/"
ref_path = os.path.join(root_path, 'JPEGImages')
anno_root = os.path.join(root_path, 'Annotations')
img_paths = os.path.join(root_path, 'ImageSets/Main/val.txt')
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

def GetAnnotBoxLoc(AnotPath):#AnotPath VOC标注文件路径
  tree = ET.ElementTree(file=AnotPath) #打开文件，解析成一棵树型结构
  root = tree.getroot()#获取树型结构的根
  ObjectSet=root.findall('object')#找到文件中所有含有object关键字的地方，这些地方含有标注目标
  ObjBndBoxSet={} #以目标类别为关键字，目标框为值组成的字典结构
  for Object in ObjectSet:
    ObjName=Object.find('name').text
    BndBox=Object.find('bndbox')
    x1 = int(BndBox.find('xmin').text)#-1 #-1是因为程序是按0作为起始位置的
    y1 = int(BndBox.find('ymin').text)#-1
    x2 = int(BndBox.find('xmax').text)#-1
    y2 = int(BndBox.find('ymax').text)#-1
    BndBoxLoc=[x1,y1,x2,y2]
    if ObjName in ObjBndBoxSet:
        ObjBndBoxSet[ObjName].append(BndBoxLoc)#如果字典结构中含有这个类别了，那么这个目标框要追加到其值的末尾
    else:
        ObjBndBoxSet[ObjName]=[BndBoxLoc]#如果字典结构中没有这个类别，那么这个目标框就直接赋值给其值吧
  return ObjBndBoxSet



def bb_iou(boxA, boxB):
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return max(iou, 0)

match = {}
for k,v in labelidx.items():
    match[v]=[]

for p,r in tqdm(zip(imglist,res)):
    img = cv2.imread(p)
    cls = r.get_field('labels').cpu().numpy()
    bbox = r.bbox
    # print(bbox)
    score = r.get_field('scores').cpu().numpy()
    ratio = get_ratio(r.size,img.shape)
    bbox = r.bbox * ratio

    anno_path = os.path.join(anno_root, os.path.basename(p).replace(".JPG", ".xml"))
    gt_boxes = GetAnnotBoxLoc(anno_path)

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

        gt_boxes_cls = gt_boxes[category]
        for bb in gt_boxes_cls:
            match_tmp = False
            for i in range(cls_max_num):
                if i<len(coord_list):
                    x1, y1, x2, y2, ss = coord_list[i,1:]
                    iou_cls = bb_iou([x1, y1, x2, y2], bb)
                    if iou_cls>=0.5: match_tmp = True
            if match_tmp:
                match[category].append(1)
            else:
                match[category].append(0)
for category in match.keys():
    print("TOP-%d ACC of %10s : %.4f"%(cls_max_num, category, np.sum(match[category])/len(match[category])))

