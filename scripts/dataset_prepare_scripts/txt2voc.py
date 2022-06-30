'''
    马拉松跌倒标注
    1.解析ADS标注结果文件，头肩信息、人体、人脸、骑手、机动车、非机动车等信息
    2.将比赛原始人体标注也整合进来
    3.将标注信息解析为VOC格式
    4.将约5000张数据，增广到10W（random crop），这里只用随机crop吧
    5.数据标注的时候，只标注了头肩，没有标注头，将头肩的x1,x2像中间靠拢1/4，y2向上靠拢1/4
    6.class: {'person', 'rider', 'non-motor', 'face', 'motor', 'head-shoulder'}:  'head-shoulder'->'head'
    data_dir: DATAs/detection/fall_v2
'''
import os
import json
import cv2
from tqdm import tqdm
from  multiprocessing import pool


headstr = """\
<annotation>
    <folder>VOC</folder>
    <filename>%s</filename>
    <source>
        <database>My Database</database>
        <annotation>COCO</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>company</name>
    </owner>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""
 
objstr = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""

tailstr = '''\
</annotation>
'''

def clip_func(i, i_min, i_max):
    if i < i_min:
        i = i_min
    if i >= i_max:
        i = i_max - 1
    return int(i)


def write_xml(anno_path, objstr, head, objs, tail, img_shape, class_list):
    print("write xml file: ", anno_path)
    w, h, c = img_shape
    with open(anno_path, 'w', encoding='utf-8') as f:
        f.write(head)
        for idx, obj in enumerate(objs): # [[x1,y1,x2,y2],[],...]
            category = class_list[idx] #'person'
            xmin, ymin, xmax, ymax = obj
            xmin = clip_func(xmin, 0, w)
            ymin = clip_func(ymin, 0, h)
            xmax = clip_func(xmax, 0, w)
            ymax = clip_func(ymax, 0, h)
            f.write(objstr % (category, xmin, ymin, xmax, ymax))
        f.write(tail)

def save_annos(img_path, anno_path, filename, objs, headstr, objstr, tailstr, class_list):
    try:
        img = cv2.imread(img_path) #h,w,c
    except:
        return

    if img is None:
        return

    if (img.shape[2] == 1):
        print(filename + " not a RGB image")
        return

    head = headstr % (filename, img.shape[1], img.shape[0], img.shape[2])
    img_shape = [img.shape[1], img.shape[0], img.shape[2]]#w,h,c
    write_xml(anno_path, objstr, head, objs, tailstr, img_shape, class_list)


def muti_processing(meta_func, task_list, num_pools = 60):
    task_pool = pool.Pool(num_pools)
    result = task_pool.map(meta_func, task_list)
    task_pool.close()
    task_pool.join()
    return result

def myProc(item):
    json_info = json.loads(item)
    img_url = json_info['image_url']
    file_name = img_url.split('/')[-1].split('.')[0]#xxx(无后缀)
    img_dir = 'JPEGImages' #os.path.split(img_url)[0].split('MMP/')[-1]
    anno_dir = 'Annotations' #img_dir.replace('images', 'voc_anno')
    # ori_labels_json_file = os.path.join(img_dir.replace('images', 'labels'), file_name+'.json')
    # person_labels = json.load(open(ori_labels_json_file))
    class_list = []
    bbox_list = []
    # for _, bbox in person_labels.items():
    #     class_list.append('person')
    #     bbox_list.append(bbox)
    if not os.path.exists('{}'.format(anno_dir)):
        os.system('mkdir -p {}'.format(anno_dir))
    anno_file_path = os.path.join(anno_dir, file_name+'.xml')
    # if not os.path.exists('{}'.format(img_dir)):
    os.system('wget -P {} {}'.format(img_dir, img_url))

    if 'sd_results' not in json_info or 'items' not in json_info['sd_results']:
        return
    labels = json_info['sd_result']['items']
    
    x1_list = []#用于统计最靠近边缘的点，用于后续的crop
    y1_list = []
    x2_list = []
    y2_list = []
    box_list = []

    img_path = os.path.join(img_dir, file_name+'.jpg')
    anno_path = anno_file_path
    if os.path.exists(anno_path):
        return
    
    for cur_label in labels:
        try:
            label_type = cur_label['labels']['属性']
            label_type = label_type.split('/')[1]
        except:
            continue
        # if label_type == 'head-shoulder':
        class_list.append(label_type)
        box_list.append(cur_label['meta']['geometry']) #x1,y1,x2,y2
        x1, y1, x2, y2 = cur_label['meta']['geometry']
        x1_list.append(x1)
        y1_list.append(y1)
        x2_list.append(x2)
        y2_list.append(y2)



    img_name = file_name + '.jpg'
    objs = box_list
    save_annos(img_path, anno_path, img_name, objs, headstr, objstr, tailstr, class_list)

def cal_class(task_list):#统计标注信息中的类别情况
    class_list = set()
    for item in tqdm(task_list):
        json_info = json.loads(item)
        try:
            labels = json_info['sd_result']['items']
            for cur_label in labels:
                label_type = cur_label['labels']['对象类型']
                class_list.add(label_type)
        except:
            continue 
    print('num: {}, class: {}'.format(len(class_list), class_list))



if __name__ == "__main__":
    # task_list = open('anno_20211011.txt').readlines()
    task_list = open('1209_dingding.txt').readlines()
    _ = muti_processing(myProc, task_list, num_pools = 200)
    # for task in task_list:
    #     myProc(task)
    # cal_class(task_list)
