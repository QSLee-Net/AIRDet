<div align="center"><img src="assets/airdet.png" width="500"></div>


## Introduction
Welcome to AIRDet! 
AIRDet is an efficiency-oriented anchor-free object detector, aims to enable robust object detection in various industry scene. With simple design, AIRDet-s outperforms series competitor e.g.(YOLOX-s, PP-YOLOE-s), and still maintains fast speed. Moreover, here you can find not only powerful models, but aslo highly efficient training strategies and complete tools from training to deployment.  

## Updates
-  **[2022/06/23: We release  AIRDet-0.0.1!]**
    * release AIRDet-series object detection models, e.g. AIRDet-s and AIRDet-m. AIRDet-s achievs mAP as 44.1% on COCO val dataset and 2.8ms latency on Nvidia-V100. AIRDet-m is a larger model build upon AIRDet-s in a heavy neck paradigm, which achieves robust improvement in detection of different object scales. For more information, please refer to [giraffe-neck](https://arxiv.org/abs/2202.04256).
    * release model convert tools for esay deployment, surppots onnx and tensorRT-fp32, TensorRT-fp16.

## Comming soon
- High efficient backbone.
- AIRDet-tiny and AIRDet-nano.
- Model distillation. 


## Model Zoo
|Model |size |mAP<sup>val<br>0.5:0.95 | Latency V100<br>TRT-FP32-BS32| Latency V100<br>TRT-FP16-BS32| FLOPs<br>(G)| weights |
| ------        |:---: | :---:     |:---:|:---: | :---: | :----: |
|[Yolox-s](./configs/yolox_s.py)   | 640 | 40.5 | 3.4 | 2.3 | 26.81 | [link]() |
|[AIRDet-s](./configs/airdet_s.py) | 640 | 44.1 | 4.4 | 2.8 | 27.56 | [link]() |
|[AIRDet-m](./configs/airdet_m.py) | 640 | 48.2 | 8.3 | 4.4 | 76.61 | [link]() |

- We reported the mAP of models on COCO2017 validation set.
- The latency in this table are measured without post-processing.

## Quick Start

<details>
<summary>Installation</summary>

Step1. Install AIRDet.
```shell
git clone https://github.com/tinyvision/AIRDet.git
cd AIRDet/
conda create -n AIRDet python=3.7 -y
conda activate AIRDet
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
export PYTHONPATH=$PWD:$PYTHONPATH
```
Step2. Install [pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip3 install cython; 
pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
</details>

<details>
<summary>Demo</summary>

Step1. Download a pretrained model from the benchmark table, e.g. airdet-s.

Step2. Use -f(config filename) to specify your detector's config. For example:
```shell
python tools/demo.py -f configs/airdet_s.py --ckpt /path/to/your/airdet_s.pth --path assets/dog.jpg
```
</details>

<details>
<summary>Reproduce our results on COCO</summary>

Step1. Prepare COCO dataset
```shell
cd <AIRDet Home>
ln -s /path/to/your/coco ./datasets/coco
```

Step 2. Reproduce our results on COCO by specifying -f(config filename)
```shell
python -m torch.distributed.launch --nproc_per_node=8 tools/train.py -f configs/airdet_s.py
```
</details>

<details>
<summary>Evaluation</summary>

```shell
python -m torch.distributed.launch --nproc_per_node=8 tools/eval.py -f configs/airdet_s.py --ckpt /path/to/your/airdet_s_ckpt.pth
```
</details>

<details>
<summary> Training on Custom Data </summary>

Step.1 Prepare your own dataset with images and labels. the directory structure should be as follow:

```shell script
BusinessVOC/
    Annotations/
        *.xml
    JPEGImages/
        *.jpg,png,PNG
    ImageSets/
        Main/
            train.txt
            test.txt
            val.txt
```

Step.2 Write the corresponding Train/Eval Dataset Path.
```shell script
self.dataset.train_ann = ("VOC_train",)
self.dataset.val_ann = ("VOC_val")
self.dataset.data_dir = 'datasets'
self.dataset.data_list = {
    "VOC_train": {
        "data_dir": "BusinessVOC/",
        "split": "train"
    },
    "VOC_val": {
        "data_dir": "BusinessVOC/",
        "split": "val"
    }, 
}

self.dataset.class2id = {
    'class_name1': 1,
    'class_name2': 2,
    'class_name3': 3,
}
self.model.head.num_classes = len(self.dataset.class2id.keys())
```

Step.3 Write your own `get_data` method:
```shell script
def get_data(self, name):
     data_dir = self.dataset.data_dir

     if name not in self.dataset.data_list:
         return None
     
     attrs = self.dataset.data_list[name]
     args = dict(
         data_dir = os.path.join(data_dir, attrs['data_dir']),
         split = attrs['split'],
         CLASS2ID = self.dataset.class2id,
     )
     return dict(
         factory="CustomVocDataset",
         args = args,
     )
```

Step.4 Put your dataset under `$LightVision_DIR/datasets`.
```shell script
ln -s /path/to/your/BusinessVOC/ ./datasets/BusinessVOC/
```

Step.5 Create your config file to control everything, including model setting, training setting, and test setting.
</details>

## Deploy

Step.1 convert torch model to onnx or trt engine, and the output file would be generated in deploy/. Note the convert mode has three options:[onnx, trt_32, trt_16].
```shell script
python tools/converter.py --output-name deploy/airdet_s.onnx -f configs/airdet_s.py -c airdet_s_ckpt.pth --inference_h 640 --inference_w 640 --mode trt_16
```

Step.2 trt engine evaluation and inference speed computation and appoint trt engine by --trt.
```shell script
python -m torch.distributed.launch --nproc_per_node=1 tools/trt_eval.py -f configs/airdet_s.py --trt deploy/airdet_s_fp16.trt --inference_h 640 --inference_w 640
```

Step.3 trt engine inference demo and appoint test image by -p.
```shell script
python tools/trt_inference.py -f configs/airdet_s.py -t deploy/airdet_s_fp16.trt -p assets/dog.jpg --input_shape 640,640
```

