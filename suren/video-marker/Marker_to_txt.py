# 2 : Convert to txt

import os, sys

try:
    from .VideoMarker import Marker
except:
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    from VideoMarker import Marker


try:
    from ...util import *
except:
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
    from util import *


data_dir = "../../data/"
file_name = data_dir + "videos/UTI/ut-interaction_set2/seq18.avi"
json_dir = data_dir + "ground_truth/UTI/ut-interaction_set2/"

# img_handle = NNHandler_image(format="mp4", img_loc=file_name)
# img_handle.runForBatch()

vid_name = file_name.replace("\\", "/").split("/")[-1].split(".")[0]
json_name = json_dir + "/%s-mask_GT.json" % vid_name
output_dir = json_dir + "/%s-mask_GT/" % vid_name

if not os.path.exists(output_dir): os.makedirs(output_dir)

js = Json(json_name)
json_data = js.read()


n_frames = json_data["frames"]
n_person = len(json_data) - 2

print("(Person, Frames) = ", n_person, n_frames)

marker = Marker()

points_2D, marked_2D, shake_2D = marker.unprocess(json_data, n_frames, n_person)

print(shake_2D.shape)

print("[n_shakes, n_frames, 4] : ", points_2D.shape)
# eprint(points)

class_name = "Mask"
# class_name = "Nomask"

for t in range(n_frames):
    txt = output_dir + '%d.txt'%(t)
    # print(txt)
    with open(txt, 'w+') as file:
        for i in range(n_person):
            if shake_2D[i][t]:
                file.write("%s "%(class_name) + " ".join(["%.3f"%x for x in points_2D[i][t]]) + "\n")




