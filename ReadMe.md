## **THE GRANA DETECTION**

Before the train of model you might create/prepare file config.py and dataset. 

1. The first iteration:

   - Dataset: '/data_hdd/projekty/023_granum/01_data/A/' - train dataset: 17 images, validation dataset: 3 images

   - Annotation: '/data_hdd/projekty/023_granum/01_data/A/A_train.json', '/scratch/projects/023/01_data/A/A_val.json'

   - Config: 'data_hdd/projekty/023_granum/03_training/AS_007_v2/config.py'

   - Weight: 'data_hdd/projekty/023_granum/03_training/AS_007_v2/latest.pth'

   - Results: bbox_mAP: 0.5650, bbox_mAP_50: 0.8720, bbox_mAP_75: 0.6160,  

     ​			 segm_mAP: 0.4640, segm_mAP_50: 0.8240, segm_mAP_75: 0.4470

2. The second iteration

   - Dataset: '/data_hdd/projekty/023_granum/01_data/zbior_A_B_nowyB' - train dataset: 41 images, validation dataset: 11 images

   - Annotation: '/data_hdd/projekty/023_granum/01_data/zbior_A_B_nowyB/train/ann_A_B_newB_train2.json', '/scratch/projects/023/01_data/zbior_A_B_nowyB/val/ann_A_B_newB_test_bezG1c.json'

   - Config: 'data_hdd/projekty/023_granum/03_training/AS_015_v2/config.py'

   - Weight: 'data_hdd/projekty/023_granum/03_training/AS_015_v2/latest.pth'

   - Results: bbox_mAP: 0.5580, bbox_mAP_50: 0.8700, bbox_mAP_75: 0.6240,  

     ​			 segm_mAP: 0.5720, segm_mAP_50: 0.8640, segm_mAP_75: 0.6070

3. Detection

   To train the model in container you can use following commands:

   python /mmdetection/tools/train.py /.../config.py --cfg-options optimizer.lr=0.004

4. Evaluation and inference 

   To inference model in container you can use following commands:

   python /mmdetection/tools/test.py /.../config.py /.../latest.pth --out /.../test.pkl --eval mAP --show

5. Visualization

   You can use Voxel51 to checking detection and prediction on images. Before visualization you have to convert detections from .pkl to .json. To convert detection use tool 'extract_dets_from_mmdet.py'

6. Tools

   - converter of detection (from file .pkl to file .json) - extract_dets_from_mmdet.py
   - converter annotations to LabelStudio - convert_to_LS.py

7. Docker

   - MMDetection repository comes with a [Dockerfile](https://github.com/open-mmlab/mmdetection/blob/master/docker/Dockerfile)

   - Already build Docker image, enriched with Tensorboard, is available here: https://hub.docker.com/r/mbuk/mmdet

