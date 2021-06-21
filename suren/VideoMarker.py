'''
USAGE

NOTE :
LMB and drag for rectangle (this signifies a shake)
RMB to remove rectangle (but this is a marking which signifies the end of shake)
MMB to unmark (do this if you click LMB or RMB by mistake)

Enter for next frame
q to quit

'''

import sys
import os
import cv2
import numpy as np
from copy import deepcopy

import bisect
from NNHandler_image import NNHandler_image

try:
    from .util import *
except:
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    from util import *

class Marker():
    color_um = (0, 255, 0)
    color_m = (255, 0, 0)
    color_s = (255, 255, 0)
    color_n = (0, 0, 255)
    thickness = 2

    def __init__(self):

        self.LM = False
        self.RM = False

        self.frame = None   # input image
        self.frame_ = None  # intermediate image
        self.img = None     # output image

        self.mark = False   # marked at current frame
        self.shake = False  # shaking or not

        self.rect = (0, 0, 0, 0)
        self.x1, self.y1, self.x2, self.y2 = 0, 0, 0, 0

        self.rect_list = []
        self.marked_list = []
        self.selected = -1

        self.mode = None

    def mark_rect_list(self):
        self.img = self.frame.copy()
        for ind, (r, m) in enumerate(zip(self.rect_list, self.marked_list)):
            rect = list(map(int, r))
            col = self.color_s if ind == self.selected else (self.color_m if m else self.color_um)
            cv2.rectangle(self.img, (rect[0], rect[1]), (rect[2], rect[3]), col, self.thickness)

    def mark_selected(self):
        iou = iou_batch(np.array(self.rect_list), self.rect)
        self.selected = np.argmax(iou) if np.max(iou) > 1e-6 else -1
        # print(iou, self.selected)

    def update_selected(self):
        if self.selected >= 0:
            self.marked_list[self.selected] = 1
            self.rect_list[self.selected] = self.rect

    def erase_selected(self):
        if self.selected > 0:
            self.marked_list[self.selected] = 0

    def mouse_callback_mark(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.LM = True
            self.mark, self.shake = True, True
            self.x1, self.y1 = x,y
            self.img = self.frame.copy()
            cv2.rectangle(self.img, (self.x1, self.y1), (x, y), self.color_um, self.thickness)
            print("LBD", self.x1, self.x2)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.LM == True:
                self.img = self.frame.copy()
                cv2.rectangle(self.img, (self.x1, self.y1), (x, y), self.color_um, self.thickness)
                # print("LBM", self.x1, self.x2, x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.LM = False
            self.x2, self.y2 = x,y
            self.img = self.frame.copy()
            cv2.rectangle(self.img, (self.x1, self.y1), (x, y), self.color_um, self.thickness)
            self.rect = (min(self.x1, x), min(self.y1, y), max(self.x1, x), max(self.y1, y))
            print("LBU", x, y)

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.mark, self.shake = True, False
            self.img = self.frame.copy()
            self.rect = (0, 0, 0, 0)
            self.x1, self.y1, self.x2, self.y2 = 0, 0, 0, 0
            print("RBU")

        elif event == cv2.EVENT_MBUTTONUP:
            self.mark, self.shake = False, False
            self.img = self.frame.copy()
            self.rect = (0, 0, 0, 0)
            self.x1, self.y1, self.x2, self.y2 = 0, 0, 0, 0

        cv2.imshow('frame', self.img)

    def mouse_callback_update(self, event, x, y, flags, param):
        if self.mode in ['u']:
            self.frame_ = self.img.copy()

        if event == cv2.EVENT_LBUTTONDOWN:
            self.LM = True
            self.mark, self.shake = True, True
            self.x1, self.y1 = x,y
            # self.img = self.frame.copy()
            # cv2.rectangle(self.img, (self.x1, self.y1), (x, y), self.color_um, self.thickness)
            print("LBD", self.x1, self.x2)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.LM == True:
                if self.mode == 'u':
                    self.img = self.frame.copy()
                    self.mark_rect_list()
                    cv2.rectangle(self.img, (self.x1, self.y1), (x, y), self.color_n, self.thickness)
                    print("LBM", self.x1, self.x2, x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.LM = False
            self.x2, self.y2 = x,y
            self.rect = (min(self.x1, x), min(self.y1, y), max(self.x1, x), max(self.y1, y))
            self.img = self.frame.copy()
            self.mark_selected()
            self.mark_rect_list()
            if self.mode == 'u':
                cv2.rectangle(self.img, (self.x1, self.y1), (x, y), self.color_n, self.thickness)
            print("LBU", x, y)

        # elif event == cv2.EVENT_RBUTTONDOWN:
        #     self.mark, self.shake = True, False
        #     self.img = self.frame.copy()
        #     self.rect = (0, 0, 0, 0)
        #     self.x1, self.y1, self.x2, self.y2 = 0, 0, 0, 0
        #     print("RBU")
        #
        # elif event == cv2.EVENT_MBUTTONUP:
        #     self.mark, self.shake = False, False
        #     self.img = self.frame.copy()
        #     self.rect = (0, 0, 0, 0)
        #     self.x1, self.y1, self.x2, self.y2 = 0, 0, 0, 0

        cv2.imshow('frame', self.img)

    def get_start_end(self, marked_points, shake):
        start = []
        end = []
        n_frames = len(shake)

        stat = 0
        for m, p in enumerate(marked_points):
            if shake[p] == 1 and stat == 0:
                stat = 1
                start.append(p)
            elif shake[p] == 0 and stat == 1:
                stat = 0
                marked_points[m] = p-1
                end.append(p-1)

        if len(start) > len(end):
            end.append(n_frames-1)
            np.append(marked_points, n_frames-1)
            # marked_points.append(n_frames-1)

        return start, end

    def process_mul(self, points_2D, marked_2D, shake_2D):
        # Can process only one person
        n_person, n_frames = marked_2D.shape

        bounding_box = {}

        for p in range(n_person):
            points = points_2D[p, :, :]
            mark = marked_2D[p, :]
            shake = shake_2D[p, :]


            marked_points = np.where(mark == 1)[0]

            start, end = self.get_start_end(marked_points, shake)
            assert len(start) == len(end) == 1, "Cannot have more than one object in a line"

            # eprint(start,end)
            sp, ep = start[0], end[0]

            s_ind = bisect.bisect_left(marked_points, sp)
            e_ind = bisect.bisect_right(marked_points, ep)
            mp_ind = marked_points[s_ind:e_ind]

            marked_rectangles = points[mp_ind, :]

            x_cords = (marked_rectangles[:, 0] + marked_rectangles[:, 2]) / 2
            y_cords = (marked_rectangles[:, 1] + marked_rectangles[:, 3]) / 2
            h = np.max(np.abs(marked_rectangles[:, 0] - marked_rectangles[:, 2]))
            w = np.max(np.abs(marked_rectangles[:, 1] - marked_rectangles[:, 3]))

            x_cords_ = np.interp(np.arange(sp, ep + 1), mp_ind, x_cords).reshape(-1, 1)
            y_cords_ = np.interp(np.arange(sp, ep + 1), mp_ind, y_cords).reshape(-1, 1)

            point_ind = np.concatenate((x_cords_ - h / 2, y_cords_ - w / 2, x_cords_ + h / 2, y_cords_ + w / 2),
                                       axis=-1)

            eprint(point_ind.shape, mp_ind.shape)

            bounding_box[p] = {
                "rectangles": point_ind.tolist(),
                "marked_points": mp_ind.tolist()
            }

        return bounding_box

    def process_single(self, points, mark, shake):
        # Can process only one person

        bounding_box = {}

        marked_points = np.where(mark == 1)[0]

        start, end = self.get_start_end(marked_points, shake)

        # eprint(start,end)

        for ind, (sp, ep) in enumerate(zip(start, end)):
            s_ind = bisect.bisect_left(marked_points, sp)
            e_ind = bisect.bisect_right(marked_points, ep)
            mp_ind = marked_points[s_ind:e_ind]

            marked_rectangles = points[mp_ind, :]

            x_cords = (marked_rectangles[:, 0] + marked_rectangles[:, 2])/2
            y_cords = (marked_rectangles[:, 1] + marked_rectangles[:, 3])/2
            # h = np.max(np.abs(marked_rectangles[:, 0] - marked_rectangles[:, 2]))
            # w = np.max(np.abs(marked_rectangles[:, 1] - marked_rectangles[:, 3]))
            h_vals = np.abs(marked_rectangles[:, 0] - marked_rectangles[:, 2])
            w_vals = np.abs(marked_rectangles[:, 1] - marked_rectangles[:, 3])

            print(x_cords.shape, y_cords.shape, h_vals.shape)

            x_cords_ = np.interp(np.arange(sp, ep+1), mp_ind, x_cords).reshape(-1, 1)
            y_cords_ = np.interp(np.arange(sp, ep+1), mp_ind, y_cords).reshape(-1, 1)
            h_vals_ = np.interp(np.arange(sp, ep+1), mp_ind, h_vals).reshape(-1, 1)
            w_vals_ = np.interp(np.arange(sp, ep+1), mp_ind, w_vals).reshape(-1, 1)


            point_ind = np.concatenate((x_cords_-h_vals_/2, y_cords_-w_vals_/2, x_cords_+h_vals_/2, y_cords_+w_vals_/2), axis = -1)

            eprint(point_ind.shape,  mp_ind.shape)

            bounding_box[ind] = {
                "rectangles" : point_ind.tolist(),
                "marked_points" : mp_ind.tolist()
            }

        return bounding_box

    def unprocess(self, json_data, n_frames, n_person):

        marked_2D = np.zeros((n_person, n_frames))
        points_2D = np.zeros((n_person, n_frames, 4))
        shake_2D = np.zeros((n_person, n_frames))

        for ind in json_data:
            if not ind.isdigit():
                continue

            s_id = int(json_data[ind]["shake_id"])
            bounding_box = json_data[ind]["bounding_box"]

            point_ind = bounding_box["rectangles"]
            mp_ind = bounding_box["marked_points"]

            sp, ep = mp_ind[0], mp_ind[-1]

            eprint(ind, sp, ep)

            points_2D[s_id, sp:ep+1, :] = np.array(point_ind)
            marked_2D[s_id, mp_ind] = 1
            shake_2D[s_id, sp:ep+1] = 1

        return points_2D, marked_2D, shake_2D


    def run(self, file_name):
        # cv2.namedWindow('frame', cv2.WINDOW_GUI_NORMAL)  # WINDOW_GUI_NORMAL stops context menu on right click
        cv2.namedWindow('frame')
        cv2.setMouseCallback('frame', self.mouse_callback_mark)

        cap = cv2.VideoCapture(file_name)
        mark = []
        shake = []
        points = []

        while (1):
            ret, frame = cap.read()
            if ret == True:
                self.mark = False
                self.frame = frame
                self.img = np.array(frame, copy=True)
                cv2.rectangle(self.img, (self.rect[0], self.rect[1]), (self.rect[2], self.rect[3]), self.color_um, self.thickness)
                cv2.imshow('frame', self.img)

                k = cv2.waitKey(0) & 0xff
                if k == ord('q'):
                    break


                points.append(self.rect)
                mark.append(1 if self.mark else 0)
                shake.append(1 if self.shake else 0)
            else:
                cv2.putText(self.img, "Reached the end, press 'q' to quit", (10, 10), 0, 0.5, (255, 255, 255), 2)
                cv2.imshow('frame', self.img)


                k = cv2.waitKey(0) & 0xff
                if k != ord('q'):
                    continue

                points[-1] = self.rect
                mark[-1] = 1 if self.mark else 0
                shake[-1] = 1 if self.shake else 0

                cv2.destroyAllWindows()
                break

        points = np.asarray(points)
        mark = np.array(mark)
        shake = np.array(shake)

        cap.release()
        cv2.destroyAllWindows()

        return points, mark, shake

    def update(self, file_name, points_2D, marked_2D, shake_2D):
        # cv2.namedWindow('frame', cv2.WINDOW_GUI_NORMAL)  # WINDOW_GUI_NORMAL stops context menu on right click
        cv2.namedWindow('frame')
        cv2.setMouseCallback('frame', self.mouse_callback_update)

        n_shakes, n_frames,  _ = points_2D.shape

        cap = cv2.VideoCapture(file_name)

        for t in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break

            self.mode = None
            self.frame = frame
            self.img = np.array(self.frame, copy=True)
            self.rect_list = points_2D[:, t, :]
            self.marked_list = marked_2D[:, t]
            self.selected = -1
            self.mark_rect_list()

            cv2.imshow('frame', self.img)

            k = cv2.waitKey(0) & 0xff

            if k == ord('d'):
                self.mode = 'd'
                while k == ord('d'):
                    cv2.putText(self.img, "Delete : Print enter to continue", (20, 20), 0, 0.75, (255, 255, 0), 2)
                    k = cv2.waitKey(0) & 0xff
                    self.erase_selected()
                    self.mark_rect_list()

                self.confirm_delete(points_2D, marked_2D, shake_2D, t)

            elif k == ord('e'):
                self.mode = 'e'
                while k == ord('e'):
                    cv2.putText(self.img, "End : Print enter to continue", (20, 20), 0, 0.75, (255, 255, 0), 2)
                    k = cv2.waitKey(0) & 0xff
                    self.erase_selected()
                    self.mark_rect_list()

                self.confirm_end(points_2D, marked_2D, shake_2D, t)


            elif k == ord('u'):
                self.mode = 'u'
                while k == ord('u'):
                    cv2.putText(self.img, "Update : Print enter to continue", (20, 20), 0, 0.75, (255, 255, 0), 2)
                    k = cv2.waitKey(0) & 0xff
                    self.update_selected()
                    self.mark_rect_list()

                self.confirm_update(points_2D, marked_2D, shake_2D, t)

            print(k)
            if k == ord('q'):
                break


            print(t, points_2D[:, t, :], marked_2D[:, t])

            # points.append(self.rect)
            # mark.append(1 if self.mark else 0)
            # shake.append(1 if self.shake else 0)

        cap.release()
        cv2.destroyAllWindows()

        return points_2D, marked_2D, shake_2D

    def confirm_delete(self, points_2D, marked_2D, shake_2D, t):
        marked_2D[self.selected, t] = 0

    def confirm_end(self, points_2D, marked_2D, shake_2D, t):
        marked_2D[self.selected, t] = 1
        shake_2D[self.selected, t] = 0


    def confirm_update(self, points_2D, marked_2D, shake_2D, t):
        marked_2D[self.selected, t] = 1
        points_2D[self.selected, t, :] = np.array(self.rect)




    def marked_video(self, file_name, points_2D, marked_2D):
        n_shakes, n_frames,  _ = points_2D.shape

        cv2.namedWindow('frame_')

        cap = cv2.VideoCapture(file_name)

        for t in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            self.frame = frame
            self.img = np.array(self.frame, copy=True)
            self.rect_list = points_2D[:, t, :]
            self.marked_list = marked_2D[:, t]
            self.selected = -1
            self.mark_rect_list()
            cv2.imshow('frame_', self.img)
            # print(rect)

            k = cv2.waitKey(0) & 0xff
            if k == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":


    UPDATE = 0
    MARK = 1

    SAVE = 1        # mark new values (0 if loaded from mem)
    DEBUG = 1
    JSON = 1        # convert and save as json
    TEST = 1



    file_name = "../data/videos/DEEE/cctv3.mp4"
    output_dir = "../data/ground_truth/DEEE/"
    raw_dir = "../data/ground_truth/DEEE/mask/"

    # img_handle = NNHandler_image(format="mp4", img_loc=file_name)
    # img_handle.runForBatch()

    if not os.path.exists(output_dir):os.makedirs(output_dir)
    vid_name = file_name.replace("\\", "/").split("/")[-1].split(".")[0]
    output_name = output_dir + "/%s-mask_GT.json" % vid_name

    js = Json(output_name)
    json_data = js.read()


    if "frames" not in json_data: json_data = {"frames": 0, "file_name": vid_name }

    n_frames = json_data["frames"]
    n_person = len(json_data) - 2

    print("(Person, Frames) = ", n_person, n_frames)

    raw_name = raw_dir + "/%s-mask_raw_%d.json"% (vid_name, n_person+1)

    js_raw = Json(raw_name)


    marker = Marker()

    if UPDATE:
        points_2D, marked_2D, shake_2D = marker.unprocess(json_data, n_frames, n_person)

        points_2D, marked_2D, shake_2D = marker.update(file_name, points_2D, marked_2D, shake_2D)

        bounding_box = marker.process_mul(points_2D, marked_2D, shake_2D)

        if JSON:
            for ind in bounding_box:
                json_data[str(ind)] = {"shake_id": ind, "bounding_box": bounding_box[ind]}

            js.update(json_data)



    elif MARK:
        # run is to mark the points
        if SAVE:
            points, mark, shake = marker.run(file_name)

            js_raw.write({'points' : points.tolist(), 'mark':mark.tolist(), 'shake':shake.tolist()})
            # np.save("points_%d.npy"%p_id, points)
            # np.save("mark_%d.npy"%p_id, mark)
            # np.save("shake_%d.npy"%p_id, shake)

        else:
            js_raw_data = js_raw.read()
            points = np.array(js_raw_data['points'])
            mark = np.array(js_raw_data['mark'])
            shake = np.array(js_raw_data['shake'])

        if DEBUG:
            print(points.shape, points)
            print(mark.shape, mark)
            print(shake.shape, shake)


        if JSON:

            # post process returns the bounding box from the marked points
            bounding_box = marker.process_single(points, mark, shake)

            # Write to json
            n_frames = max(n_frames, len(mark))
            json_data["frames"] = n_frames

            for ind in bounding_box:
                json_data[str(n_person+ind)] = {"shake_id": n_person+ind, "bounding_box": bounding_box[ind]}

            n_person = len(json_data) - 2

            js.update(json_data)



    if TEST:

        points_2D, marked_2D, _ = marker.unprocess(json_data, n_frames, n_person)

        print("[n_shakes, n_frames, 4] : ", points_2D.shape)
        # eprint(points)

        marker.marked_video(file_name, points_2D, marked_2D)



