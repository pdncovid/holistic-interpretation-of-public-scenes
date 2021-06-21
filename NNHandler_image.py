import numpy as np
import os
import json
import cv2
from glob import glob

from NNHandler import NNHandler
from suren.util import get_iou, Json, eprint


class NNHandler_image(NNHandler):
    VID_FORMAT = ["avi", "mp4", "ts"]
    IMG_FORMAT = ["jpg", "png"]


    def __init__(self, format, img_loc=None, json_file=None):

        super().__init__()

        print("Creating an Image handler")
        self.format = format
        self.img_loc = img_loc
        self.json_file = json_file

        self.cap = None
        self.json_data = None
        self.time_series_length = None

        self.width = None
        self.height = None
        self.fps = None

        self.vid_out = None

        print(self)

    def __repr__(self):
        lines = []
        if self.img_loc is not None:
            lines.append("\t[*] Image location : %s"%self.img_loc)
        if self.json_file is not None:
            lines.append("\t[*] Json location : %s" % self.json_file)
        if self.time_series_length is not None:
            lines.append("\t[*] Frames : {}".format(self.time_series_length))
        if self.width is not None and self.height is not None:
            lines.append("\t[*] (h, w) : ({}, {})".format(self.height, self.width))
        if self.fps is not None:
            lines.append("\t[*] FPS : {}".format(self.fps))

        return "\n".join(lines)

    def count_frames(self, path=None):
        if self.format in NNHandler_image.VID_FORMAT:
            if path is None: path = self.img_loc

            cap = cv2.VideoCapture(path)
            # self.init_param(cap=cap)

            total = 0
            # loop over the frames of the video
            while True:
                (grabbed, frame) = cap.read()
                if not grabbed:
                    break
                total += 1

            cap.release()

            return total

        elif self.format in NNHandler_image.IMG_FORMAT:
            if path is None: path = self.img_loc

            img_names = list(map(lambda x: x.replace("\\", "/"), glob(path + "/*.%s" % format)))
            return len(img_names)

        else:
            raise NotImplementedError

    def open(self, init_param = False):
        if self.format in NNHandler_image.VID_FORMAT:
            self.cap = cv2.VideoCapture(self.img_loc)
            eprint("Frames will only be read linearly")

            if init_param: self.init_param()

    def init_param(self, cap=None):
        if cap is None: cap = self.cap
        assert cap is not None, "Capture cannot be none"

        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.time_series_length = int(self.n_frames/self.fps)

    def close(self):
        if self.format in NNHandler_image.VID_FORMAT:
            self.cap.release()

    def init_writer(self, out_name, h, w, fps=30, encoding="XVID"):
        fourcc = cv2.VideoWriter_fourcc(*'%s'%encoding)
        self.vid_out = cv2.VideoWriter(out_name, fourcc, fps, (w, h))

    def write_frame(self,frame):
        assert self.vid_out is not None, "Initialize writer by calling init_writer(out_name, h, w, fps)"
        self.vid_out.write(frame)

    def close_writer(self):
        self.vid_out.release()

    def read_frame(self, frame_no=None):
        if self.format in NNHandler_image.VID_FORMAT:
            # raise NotImplementedError("Don't do this. It causes errors")
            # self.cap.set(1, frame - 1)
            res, frame = self.cap.read()

            return frame

        elif self.format in NNHandler_image.IMG_FORMAT:
            return cv2.imread(self.json_data[str(frame_no)])

    def init_from_img_loc(self, img_loc=None, show=False):
        img_loc = self.img_loc if img_loc is None else img_loc
        assert os.path.exists(img_loc), "Source image/video does not exist in path : %s"%img_loc

        if self.format in NNHandler_image.VID_FORMAT:

            self.time_series_length = self.count_frames(img_loc)
            self.json_data = None

            if show: self.show()

            return self.json_data

        elif self.format in NNHandler_image.IMG_FORMAT:

            img_names = list(map(lambda x: x.replace("\\", "/"), glob(img_loc + "/*.%s" % format)))
            n_frames = len(img_names)

            self.time_series_length = n_frames
            self.json_data = {i: img for (i, img) in enumerate(img_names)}

            if show: self.show()

            return self.json_data

        else:
            raise NotImplementedError

    def init_from_json(self, json_file=None, show=False):
        json_file = self.json_file if json_file is None else json_file
        assert os.path.exists(json_file), "Json file does not exist in path : %s"%json_file

        if self.format in NNHandler_image.VID_FORMAT:
            raise Exception("No json for Videos : %s" % self.format)

        elif self.format in NNHandler_image.IMG_FORMAT:
            json_file = self.json_file if json_file is None else json_file

            with open(json_file) as json_file:
                data = json.load(json_file)

            self.time_series_length = data.pop("frames")
            self.json_data = data

            if show: self.show()

            return self.json_data

        else:
            raise NotImplementedError

    def show(self):

        WAIT = 20

        cv2.namedWindow('rgb')
        self.open()

        for i in range(self.time_series_length):
            rgb = self.read_frame(i)

            cv2.imshow('rgb', rgb)

            if cv2.waitKey(WAIT) & 0xff == ord('q'): break

        cv2.destroyAllWindows()
        self.close()

    def write_json(self, json_file=None):
        json_file = self.json_file if json_file is None else json_file

        if self.format in NNHandler_image.VID_FORMAT:
            eprint("Cannot create json from Videos : %s" % format)

        elif self.format in NNHandler_image.IMG_FORMAT:
            js = Json(json_file)
            dic = {"frames": self.time_series_length}
            for i in self.json_data:
                dic[i] = self.json_data[i]

            js.write(dic)

        else:
            raise NotImplementedError

    def runForBatch(self):
        if self.img_loc is None and self.json_file is None: raise ValueError("Both img_loc and json_file cannot be None")

        elif self.img_loc is None and self.json_file is not None: self.init_from_json()

        else:
            self.init_from_img_loc()
            if self.json_file is not None: self.write_json()

        print("\t[f] Read %s file with %d frames" %(self.format, self.time_series_length))


if __name__ == "__main__":

    vid_loc = "./suren/temp/18.avi"  # Change this to your video directory
    # img_loc = "./suren/temp/rgb"            # Change this to your image directory
    # json_file = "./data/vid-01-image.json"   # Rename this too... if necessary

    img_handle = NNHandler_image(format="avi", img_loc=vid_loc)
    img_handle.runForBatch()
    img_handle.show()
