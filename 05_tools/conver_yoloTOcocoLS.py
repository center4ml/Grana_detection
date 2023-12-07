import json
import glob
import argparse

import pandas as pd
import numpy as np
from skimage import measure
import cv2
import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description='convert yolo results to coco Label Studio format')
    parser.add_argument('--path_to_predict', help='path of the predict folder')
    parser.add_argument('--out', help='path of the output file')
    args = parser.parse_args()

    return args

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

def converter_to_LabelStudio(path_to_predict):
    list_lab = [x for x in glob.glob(f'{path_to_predict}/labels/*')]

    lb_list = []

    for lab in list_lab:
        img = f'{lab}'.split("/")[-1].replace('txt', 'png')
        im = cv2.imread(f'{path_to_predict}/{img}')

        data = pd.read_csv(f'{lab}', sep="\t", header=None)
        data['list'] = [x.split(" ")[1:-1] for x in data[0]]
        data['float'] = data['list'].apply(lambda x: [float(i) for i in x])
        data['conf'] = data[0].apply(lambda x: x.split(" ")[-1:])
        data['points_yolo'] = data['float'].apply(lambda x: [x[i:i + 2] for i in range(0, len(x), 2)])

        data['original_width'] = im.shape[1]
        data['original_height'] = im.shape[0]
        data['image_rotation'] = 0
        data['from_name'] = "label"
        data['to_name'] = "image"
        data['type'] = "polygonlabels"
        data['id'] = [x for x in range(len(data))]

        data['points'] = [np.column_stack((np.array(data.points_yolo[x])[:, 0] * data.original_width[x],
                                           np.array(data.points_yolo[x])[:, -1] * data.original_height[x])) for x in
                          range(len(data))]

        value_true = data[['points']]
        value_dict = value_true.to_dict(orient='records')

        data['value'] = [x for x in value_dict]

        result = data[
            ['original_width', 'original_height', 'image_rotation', 'value', 'id', 'from_name', 'to_name', 'type']]
        result_lb = result.to_dict(orient='records')

        pred_v = {"model_version": "one", "score": 0.95, "result": result_lb}
        pred = []
        pred.append(pred_v)
        images = {"images": img}

        ls = {'data': images, 'predictions': pred}
        lb_list.append(ls)

    return lb_list

def main():
    args = parse_args()
    ann_LS = converter_to_LabelStudio(args.path_to_predict)

    with open(args.out, 'w') as f:
        json.dump(ann_LS, f, cls=NumpyEncoder)

# %% run main
if __name__ == '__main__':
    main()