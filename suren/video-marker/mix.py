# 3: Mix images, and rename

import sys, os
import shutil
import cv2

try:
    from ....NNHandler_image import NNHandler_image
    from ...util import *

except:
    file_dir = os.path.dirname(os.path.realpath(__file__))
    proj_dir = os.path.join(file_dir, os.pardir)

    sys.path.append(file_dir)
    sys.path.append(proj_dir)

    from NNHandler_image import NNHandler_image
    from util import *


data_dir = "../../data/"
file_name = data_dir + "videos/UTI/ut-interaction_set2/seq18.avi"

vid_name = file_name.replace("\\", "/").split("/")[-1].split(".")[0]

txt_dir = data_dir + "ground_truth/UTI/ut-interaction_set2/%s-mask_GT/" %vid_name

img_handle = NNHandler_image(format="avi", img_loc=file_name)
img_handle.runForBatch()
img_handle.open()

class_name = "Nomask"
output_dir = data_dir + "ground_truth/temp/" + class_name + "/"
# print(output_dir)

if not os.path.exists(output_dir): os.makedirs(output_dir)

for t in range(img_handle.time_series_length):
    img = img_handle.read_frame(t)
    txt = txt_dir + "%d.txt"%t

    # if not os.path.exists(txt): input("ERROR %s: Press???"%txt)

    shutil.copy(txt, output_dir + "Label/%s-%d.txt"%(vid_name, t))
    cv2.imwrite(output_dir + "%s-%d.png"%(vid_name, t), img)




