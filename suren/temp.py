import configparser
import cv2
import os
# from Visualizer import Visualizer
from NNHandler_image import NNHandler_image
def read_ini(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    for section in config.sections():
        print(section)
        for key in config[section]:
            print(section, (key, config[section][key]))

# file = "../data/temp_files/oxford.ini"
# read_ini(file)

# print(os.path.exists(file))

# inp = "../data/videos/DEEE/cctv3.mp4"
# inp = "../data/videos/DEEE/cctv5.mp4"

img_handle = NNHandler_image(format="avi", img_loc=inp)
img_handle.open()

for t in range(1000):

    rgb = img_handle.read_frame(t)
    # cv2.imshow('frame', rgb)
    cv2.imwrite("{}inp-{:04d}.jpg".format('../suren/temp/DEEE/', t), rgb)

    # k = cv2.waitKey(20)

    # if k & 0xff == ord('q'): break

