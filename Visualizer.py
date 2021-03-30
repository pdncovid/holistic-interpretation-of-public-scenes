import cv2
import numpy as np
from collections import defaultdict
import argparse

from Graph import Graph
from NNHandler_handshake import NNHandler_handshake
from NNHandler_image import NNHandler_image
from NNHandler_yolo import NNHandler_yolo

from suren.util import eprint, stop, progress, Json
try:
    # import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
except ImportError as e:
    print(e)


class Visualizer:
    def __init__(self, graph=None, yolo=None, handshake=None, img=None):
        """

        :param graph: Graph
        :param yolo: NNHandler_yolo
        :param handshake: NNHandler_handshake
        :param img: NNHandler_image
        """
        self.graph = graph
        self.yolo_handle = yolo
        self.hs_handle = handshake
        self.img_handle = img

    def plot(self, WAIT=20):

        if Graph.plot_import() is not None:
            eprint("Package not installed", Graph.plot_import())
            return

        if self.graph is not None:
            cmap = self.graph.get_cmap(show=True)
            sc_x, sc_y, lines = self.graph.get_plot_points()

            # print(sc_x.shape, sc_y.shape, cmap.shape)

            print(cmap.shape)

            cmap_ = np.array(cmap[:, -1, :-1] * 255)[:, [2, 1, 0]]
            # cmap_ = cv2.cvtColor(cmap_.reshape(1, -1, 3), cv2.COLOR_RGB2BGR).reshape(-1, 3)
            print(cmap_)

            for n, p in enumerate(self.graph.nodes):
                p.params["col"] = cmap_[n]

        # PLOT
        cv2.namedWindow("plot")



        # plt.figure()
        # plt.scatter(np.arange(4), np.ones(4), color=cmap_)
        # plt.show()

        img_handle.open()
        for t in range(self.graph.time_series_length):
            rgb = img_handle.read_frame(t)
            rgb_ = rgb.copy()

            if self.graph is not None:
                sc_x_ = list(map(int, sc_x[:, t]))
                sc_y_ = list(map(int, sc_y[:, t]))

                for p in range(len(sc_x_)):
                    cv2.circle(rgb_, (sc_x_[p], sc_y_[p]), 1, tuple(cmap_[p]), 5)

                for l in lines[t]:
                    cv2.line(rgb_, tuple(np.array(l[:, 0]).astype(int)), tuple(np.array(l[:, 1]).astype(int)),
                             (255, 255, 255), 3)

            if self.yolo_handle is not None:
                for p in self.graph.nodes:
                    x_min, y_min, x_max, y_max = map(int, [p.params["xMin"][t], p.params["yMin"][t], p.params["xMax"][t], p.params["yMax"][t]])
                    cv2.rectangle(rgb_, (x_min, y_min), (x_max, y_max), p.params["col"], 2)

            if self.hs_handle is not None:
                if str(t) in self.hs_handle.json_data:
                    if self.hs_handle.is_tracked:
                        bb_dic = self.hs_handle.json_data[str(t)]
                    else:
                        bb_dic = self.hs_handle.json_data[str(t)]["bboxes"]
                    for bbox in bb_dic:
                        x_min, x_max, y_min, y_max = map(int, [bbox["x1"], bbox["x2"], bbox["y1"], bbox["y2"]])
                        cv2.rectangle(rgb_, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)


            if (t + 1) % 20 == 0:
                progress(t + 1, self.graph.time_series_length, "drawing graph")

            # fig.canvas.draw()
            #
            # # convert canvas to image
            # img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            # img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            # img = img.reshape(rgb.shape[::-1] + (3,))

            # img is rgb, convert to opencv's default bgr
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # img_sctr = img.copy()
            # print(img_sctr.shape, rgb.shape)

            # img_ = np.hstack((img, img_sctr))

            # display image with opencv or any operation you like
            cv2.imshow("plot", rgb_)

            k = cv2.waitKey(WAIT) & 0xFF
            if k == 'q':
                break
        img_handle.close()
    # cap.release()

    # plt.show(block=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--graph_file","-g",type=str,dest="graph_file",default='./data/vid-01-graph.json')
    parser.add_argument("--nnout_yolo","-y",type=str,dest="nnout_yolo",default='./data/vid-01-yolo.txt')
    parser.add_argument("--nnout_handshake","-hs",type=str,dest="nnout_handshake",default='./data/vid-01-handshake_track.json')
    parser.add_argument("--video_file","-v",type=str,dest="video_file",default='./suren/temp/seq18.avi')
    # parser.add_argument("--config_file","-c",type=str,dest="config_file",default="args/visualizer-01.json")

    args = parser.parse_args()

    # if None in [args.nnout_yolo, args.nnout_handshake,args.video_file,args.graph_file]:
    #     print("Running from config file")
    #
    #     with open(args.config_file) as json_file:
    #         data = json.load(json_file)
    #         args = argparse.Namespace()
    #         dic = vars(args)
    #
    #         for k in data.keys():
    #             dic[k]=data[k]
    #         print(args)


    g = Graph()



    yolo_handler = NNHandler_yolo(args.nnout_yolo)
    # yolo_handler.connectToGraph(g)
    # yolo_handler.runForBatch()
    g.init_from_json(args.graph_file)

    # hs_handler = NNHandler_handshake(args.nnout_handshake, is_tracked=True)
    hs_handler = NNHandler_handshake('./data/vid-01-handshake.json', is_tracked=False)        # This is without DSORT tracker and avg
    # hs_handler = NNHandler_handshake('./data/vid-01-handshake_track.json', is_tracked=True)       # With DSORT and avg
    hs_handler.connectToGraph(g)
    hs_handler.runForBatch()

    img_handle = NNHandler_image(format="avi", img_loc=args.video_file)
    # img_handle = NNHandler_image(format="avi", img_loc="./suren/temp/seq18.avi")
    img_handle.runForBatch()

    vis = Visualizer(graph= g, yolo=None, handshake=None, img=img_handle)
    vis.plot(WAIT=25)

