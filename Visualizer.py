import cv2
import numpy as np
from collections import defaultdict

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
        self.graph = graph
        self.yolo_handle = yolo
        self.hs_handle = handshake
        self.img_handle = img

    def plot(self):

        def get_cmap(show=False):
            colors = cm.hsv(np.linspace(0, .8, self.n_nodes))
            window = 10

            col_arr = np.ones((window, 4))

            col_arr[:, -1] = np.power(.8, np.arange(window))[::-1]

            arr1 = np.tile(colors, (window, 1, 1)).transpose((1, 0, 2))
            # print(colors.shape, arr1.shape)
            arr2 = np.tile(col_arr, (self.n_nodes, 1, 1))
            # print(col_arr.shape, arr2.shape)

            cmap = arr1 * arr2

            # print(arr1[1, :, :], arr2[1, :, :])

            # print(colors)

            # stop()
            if show:
                x = np.tile(np.arange(cmap.shape[0]), (cmap.shape[1], 1))
                y = np.tile(np.arange(cmap.shape[1]), (cmap.shape[0], 1)).transpose()
                # print(x)
                # print(y)
                plt.figure()
                plt.title("Colour map (Close to continue)")
                plt.scatter(x.flatten(), y.flatten(), color=np.reshape(cmap, (-1, 4), order='F'))
                plt.show()

            return cmap

        if Graph.plot_import() is not None:
            eprint("Package not installed", Graph.plot_import())
            return

        if self.graph is not None:
            cmap = self.graph.get_cmap(show=True)
            sc_x, sc_y, lines = self.graph.get_plot_points()

            print(sc_x.shape, sc_y.shape, cmap.shape)

            print(cmap.shape)

            cmap_ = np.array(cmap[:, -1, :-1] * 255)[:, [2, 1, 0]]
            # cmap_ = cv2.cvtColor(cmap_.reshape(1, -1, 3), cv2.COLOR_RGB2BGR).reshape(-1, 3)
            print(cmap_)

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
                    print(l)
                    cv2.line(rgb_, tuple(np.array(l[:, 0]).astype(int)), tuple(np.array(l[:, 1]).astype(int)),
                             (255, 255, 255), 3)

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

            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
        img_handle.close()
    # cap.release()

    # plt.show(block=True)


if __name__ == "__main__":
    g = Graph()

    # yolo_handler = NNHandler_yolo('./data/vid-01-graph.json')
    # yolo_handler.connectToGraph(g)
    # yolo_handler.runForBatch()
    g.init_from_json('./data/vid-01-graph.json')

    hs_handler = NNHandler_handshake('./data/vid-01-handshake.json')
    hs_handler.connectToGraph(g)
    hs_handler.runForBatch()

    img_handle = NNHandler_image(format="avi", img_loc="./suren/temp/seq18.avi")
    img_handle.init_from_json()

    vis = Visualizer(graph= g, yolo=None, handshake=hs_handler, img=img_handle)
    vis.plot()





