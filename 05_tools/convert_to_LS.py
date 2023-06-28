import json

import pandas as pd
import numpy as np
from skimage import measure
import argparse

import pycocotools.mask as mask_utils
import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description='convert mmdetection json results to Label Studio format')
    parser.add_argument('detections_file', help='path of the json detection file')
    parser.add_argument('--coco', help='path of the coco annotations')
    parser.add_argument('--out', help='path of the output file')
    args = parser.parse_args()

    return args

input_file = '/scratch/projects/023/04_tools/inference/wd/r.json'
annotations_file = '/scratch/projects/023/04_tools/inference/empty.json'
output_file = '/scratch/projects/023/04_tools/inference/wd/r_LS.json'

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def get_longest(contours):
    lengths = [np.linalg.norm(np.diff(c, axis=0), axis=1).sum() for c in contours]
    return contours[np.argmax(lengths)]
  
def convert_rle_to_polygon(rle_mask_dict, max_edges=20): #max_edges - ilość punktów poligonu
    binary_mask = mask_utils.decode(rle_mask_dict)
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    if len(contours) > 1:
        contour = get_longest(contours)
    else:
        contour = contours[0]

    contour -= 1 # correct for padding
    contour = close_contour(contour)

    for tol in np.arange(0.5, 5, 0.1):
        polygon = measure.approximate_polygon(contour, tol)
        n_gons = len(polygon)
        if n_gons <= max_edges:
            break
    
    polygon = np.flip(polygon, axis=1)
    # after padding and subtracting 1 we may get -0.5 points in our polygon. Replace it with 0
    polygon = np.where(polygon>=0, polygon, 0)
    # segmentation = polygon.ravel().tolist()

    return polygon

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

    # to jest funkcja do generowania formatu coco do Label Studio 
# (potrzebuję do tego json z mmdetection i pusty json)
def converter_to_LabelStudio(detections_path, annotations_path, min_confidence=0.3):
   
    annotations = json.load(open(annotations_path))
    input_data = json.load(open(detections_path))
    if len(annotations['images']) != len(input_data):
        raise ValueError(f"{len(annotations['images'])} images in COCO while {len(input_data)} in detections file")
    
    lb_list=[]

    df = pd.DataFrame(annotations['images'])
    
    for x in tqdm.tqdm(range(len(df))):
        img={'images':df['file_name'][x].split('/')[-1]}
        df_ann= pd.DataFrame(input_data[x])
        if len(df_ann) == 0:
            lb_list.append({'data': img, 'predictions': []})
            continue
            
        df_ann = df_ann[df_ann.confidence >= min_confidence]
        
        if len(df_ann) == 0:
            lb_list.append({'data': img, 'predictions': []})
            continue
        
        size= pd.DataFrame(df_ann['mask'][0])
        height=size['size'][0]
        width=size['size'][1]
        
        df_ann['points']=df_ann['mask'].apply(convert_rle_to_polygon)
        df_ann['original_width']=width
        df_ann['original_height']=height
        df_ann['trueP']=[
            np.column_stack((
                df_ann.points[x][:,0]/df_ann.original_width[x]*100, 
                df_ann.points[x][:,-1]/df_ann.original_height[x]*100
            )) for x in range(len(df_ann))
        ]
        
        value_test_dict = df_ann[['trueP']].rename(columns={"trueP": "points"}).to_dict(orient='records')
        # value_true=df_ann[['trueP']]
        # value_true.rename(columns={"trueP": "points"}, inplace=True)
        # value_test_dict=value_true.to_dict(orient='records')
        
        df_ann['value_dic']=[x for x in value_test_dict]
        df_ann['image_rotation']=0
        df_ann['from_name']="label"
        df_ann['to_name']="image"
        df_ann['type']="polygonlabels"
        df_ann['id']= [x for x in range(len(df_ann))]
        df_ann.rename(columns={"value_dic": "value"}, inplace=True)
        
        result=df_ann[['original_width', 'original_height', 'image_rotation', 'value', 'id', 'from_name', 'to_name', 'type']]
        result_lb=result.to_dict(orient='records')
        
        pred_v={"model_version": "one", "score": 0.95, "result": result_lb}
        pred=[]
        pred.append(pred_v)
        
        ls={'data': img, 'predictions': pred}
        
        lb_list.append(ls)
        
    return lb_list

def main():
    args = parse_args()
    all_imgs = converter_to_LabelStudio(args.detections_file, args.coco)

    with open(args.out, 'w') as f:
        json.dump(all_imgs, f, cls=NumpyEncoder)

# %% run main
if __name__ == '__main__':
    main()