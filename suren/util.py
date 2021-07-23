from __future__ import print_function

import json
import configparser

import os
import sys
import numpy as np

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

class Json():

    def __init__(self, name, OW=False, verbose=True):
        self.name = name

        if os.path.exists(name) and OW==False:
            if verbose: print("json file exists")
        else:
            if verbose: print("creating json file")
            self.write({})

    def write(self, data):
        Json.is_jsonable(data)  # Check if data is serializable

        with open(self.name, 'w+') as outfile:
            json.dump(data, outfile, indent=4)


    def read(self):
        with open(self.name, 'r') as file:
            if len(file.readlines()) != 0: file.seek(0)
            data = json.load(file)
        return data


    def update(self, data):
        temp = self.read()
        temp.update(data)
        self.write(temp)


    def read_data(self, attr):
        data = self.read()
        return data[attr]


    @staticmethod
    def is_jsonable(x):
        try:
            json.dumps(x)
        except Exception as e:
            raise Exception(e)

def get_iou(bb1, bb2, mode=0):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : list
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : list
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    bb1[0], bb1[2] = sorted([bb1[0], bb1[2]])
    bb1[1], bb1[3] = sorted([bb1[1], bb1[3]])
    bb2[0], bb2[2] = sorted([bb2[0], bb2[2]])
    bb2[1], bb2[3] = sorted([bb2[1], bb2[3]])

    # print(bb1, bb2)

    assert bb1[0] <= bb1[2], "bb1[0] < bb1[2], (%d, %d)"%(bb1[0], bb1[2])
    assert bb1[1] <= bb1[3], "bb1[1] < bb1[3], (%d, %d)"%(bb1[1], bb1[3])
    assert bb2[0] <= bb2[2], "bb2[0] < bb2[2], (%d, %d)"%(bb2[0], bb2[2])
    assert bb2[1] <= bb2[3], "bb2[1] < bb2[3], (%d, %d)"%(bb2[1], bb2[3])

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    # print(x_right, x_left, y_bottom, y_top)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    if mode:
        iou = intersection_area / float(bb1_area)
    else:
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def stop():
    input("Enter to continue")

def progress(count, total, status=''):
    # bar_len = 60
    # filled_len = int(round(bar_len * count / float(total)))
    # bar = '=' * filled_len + '-' * (bar_len - filled_len)

    percents = round(100.0 * count / float(total), 1)

    if count == total:
        print('%s%s ...%s' % (percents, '%', status), end='\n')
    else:
        print('%s%s ...%s' % (percents, '%', status), end='\n')


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return (o)




def read_ini(file_path, config_json):
    config = configparser.ConfigParser()
    config.read(file_path)
    for section in config.sections():
        for key in config[section]:
            print((key, config[section][key]))

