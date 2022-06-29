# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import os,sys,cv2
import pdb
import argparse
import xml.etree.ElementTree as ET
from tqdm import tqdm


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


if __name__ == "__main__":
    # electric
    cls_max_num = 1
    root_path = "./datasets/VOCelectric/"
    origin_root = root_path + "JPEGImages"
    visualization_anno = "./showcases/cls3/vis_anno"
    os.makedirs(visualization_anno, exist_ok=True)
    anno_root = root_path + "Annotations"
    visualization_txt = root_path + "ImageSets/Main/val.txt"
    images_name = os.listdir(origin_root)
    with open(visualization_txt, "r") as fr:
        images_name=fr.readlines()
    images_name = ["%s.JPG"%(image_name.split("\n")[0]) for image_name in images_name]

    images_name = [image_name for image_name in images_name if "JPG" in image_name]
    color_dict = {"hammer":(0,255,0), "insulation":(255,0,0), "dowel":(0,0,255)}


    for idx, image_name in enumerate(tqdm(images_name)):
        image_path = os.path.join(origin_root, image_name)
        des_image_path = os.path.join(visualization_anno, image_name)
        anno_path = os.path.join(anno_root, image_name.replace(".JPG", ".xml"))

        img_np = cv2.imread(image_path)
        try:
            target = GetAnnotBoxLoc(anno_path)

            for class_id in target.keys():
                if class_id not in image_name:
                    continue
                coord_list = target[class_id]
                color = color_dict[class_id]
                for bb in coord_list:
                    img_np = cv2.rectangle(img_np, (bb[0],bb[1]), (bb[2],bb[3]), (10,10,10), 4)

            cv2.imwrite(des_image_path, img_np)
        except:
            print("%s maybe without xml"%(image_name))
