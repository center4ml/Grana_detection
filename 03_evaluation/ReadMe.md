## **The Evaluation and inference **

 

1. Evaluation and inference 

   To inference model in container you can use following commands:

   python /mmdetection/tools/test.py /.../config.py /.../latest.pth --out /.../test.pkl --eval mAP --show

   

   The first iteration: python /mmdetection/tools/test.py \
   
   /scratch/projekty/023_granum/03_training/AS_007_v2/config.py \
   
   /scratch/projekty/023_granum/03_training/AS_007_v2/latest.pth --out  \ 
   
   /scratch/projekty/023_granum/03_training/AS_007_v2/resultsA_007_v2.pkl --eval bbox segm --show
   
   Results: bbox_mAP: 0.565, bbox_mAP_50: 0.872, bbox_mAP_75: 0.616,  
   
   ​			 segm_mAP: 0.464, segm_mAP_50: 0.824, segm_mAP_75: 0.447
   
   
   
   The second iteration: python /mmdetection/tools/test.py \
   
   /scratch/projekty/023_granum/03_training/AS_015_v2/config.py \
   
   /scratch/projekty/023_granum/03_training/AS_015_v2/latest.pth --out  \ 
   
   /scratch/projekty/023_granum/03_training/AS_015_v2/resultsABnewB_015_v2.pkl --eval bbox segm --show
   
   Results: bbox_mAP: ..., bbox_mAP_50: ..., bbox_mAP_75: ...,  
   
   ​			 segm_mAP: ...., segm_mAP_50: ..., segm_mAP_75: ...

​	

​	
