import os

import cv2
from NNHandler_mask import NNHandler_mask
from NNHandler_image import NNHandler_image
from suren.util import Json
from Visualizer import Visualizer


def json_to_text(js : Json, img_handle : NNHandler_image, out_dir : str, mask : bool = None):

    data_dir = out_dir + "/data"
    label_dir = out_dir + "/label"

    if not os.path.exists(data_dir): os.mkdir(data_dir)
    if not os.path.exists(label_dir): os.mkdir(label_dir)

    data = js.read()

    img_handle.open()
    for t in range(img_handle.time_series_length):
        img =img_handle.read_frame(t)

        dic = data[str(t)]

        with open(label_dir + "/%s.txt" %t, 'w+') as file:
            for p in dic:
                x1, y1, x2, y2 = p["x1"], p["y1"], p["x2"], p["y2"]
                if mask is None:
                    mask = p["mask"]

                cls = 0 if mask else 1

                file.write("{} {} {} {} {}\n".format(cls, x1, y1, x2, y2))

        cv2.imwrite(data_dir + "/%s.png"%t, img)

    img_handle.close()

dataset = "DEEE"
vid = "cctv2.mp4"

img_loc = "./data/videos/{}/{}".format(dataset, vid)
json_loc = "./data/labels/{}/mask/{}-mask.json".format(dataset, vid.split('.')[0])

visualize = True
verbose = True
tracker = False
overwrite = False

# TEST
img_handle = NNHandler_image(format="avi", img_loc=img_loc)
img_handle.runForBatch()

# NNHandler_mask.weigths_filename = './checkpoints/yolov4-obj_best'

nn_handle = NNHandler_mask(mask_file=json_loc, is_tracked=tracker)

# nn_handle.weigths_filename = './checkpoints/yolov4-obj_best'


if os.path.exists(json_loc) and not overwrite:
    nn_handle.init_from_json()
else:
    nn_handle.create_yolo(img_handle)
    nn_handle.save_json()

vis = Visualizer(person=nn_handle, img=img_handle)
vis.init_vid(vid_show=True)
vis.plot(WAIT=0)


# js = Json(json_loc)
# json_to_text(js, img_handle, "./suren/temp/18", mask=False)

