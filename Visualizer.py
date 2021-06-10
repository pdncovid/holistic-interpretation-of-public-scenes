import cv2
import numpy as np
from collections import defaultdict
import argparse
import json

from Graph import Graph
from NNHandler_handshake import NNHandler_handshake
from NNHandler_image import NNHandler_image
from NNHandler_person import NNHandler_person
from NNHandler_openpose import NNHandler_openpose

from suren.util import eprint, progress, Json

try:
    # import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
except ImportError as e:
    print(e)


class Visualizer:
    def __init__(self, graph=None, person=None, handshake=None, img=None, openpose=None, out_name=None):
        self.graph = graph
        self.person_handle = person
        self.hs_handle = handshake
        self.img_handle = img
        self.openpose_handle = openpose

        self.out_name = out_name

        # Scatter plot components
        self.plot_network = False
        self.network_scatter = False
        self.network_lines = False

        # Img/Video components
        self.plot_vid = False
        self.vid_bbox = False
        self.vid_hbox = False
        self.vid_scatter = False
        self.vid_lines = False
        self.vid_keypoints = False

    def init_network(self, network_scatter=True, network_lines=True):
        self.plot_network = True
        self.network_scatter = network_scatter
        self.network_lines = network_lines

    def init_vid(self, vid_bbox=True, vid_hbox=True, vid_scatter=True, vid_lines=True, vid_keypoints=True):
        self.plot_vid = True
        self.vid_bbox = vid_bbox
        self.vid_hbox = vid_hbox
        self.vid_scatter = vid_scatter
        self.vid_lines = vid_lines
        self.vid_keypoints = vid_keypoints

    def plot(self, WAIT=20, show_cmap=True):

        if Graph.plot_import() is not None:
            eprint("Package not installed", Graph.plot_import())
            return

        # Process and get all graph points till time t
        if self.graph is not None:
            # colour map
            cmap = self.graph.get_cmap(show=show_cmap)
            # scatter x, y and lines
            sc_x, sc_y, lines = self.graph.get_plot_points()

            # print(sc_x.shape, sc_y.shape, cmap.shape)

            ylim = [np.min(sc_y, axis=None) - 5, np.max(sc_y, axis=None) + 5]
            xlim = [np.min(sc_x, axis=None) - 5, np.max(sc_x, axis=None) + 5]

            print(cmap.shape)

            cmap_vid = np.array(cmap[:, -1, :-1] * 255)[:, [2, 1, 0]]
            cmap_network = np.reshape(cmap[:, -1, :], (-1, 4), order='C')
            # cmap_ = cv2.cvtColor(cmap_.reshape(1, -1, 3), cv2.COLOR_RGB2BGR).reshape(-1, 3)
            print(cmap_vid)

            for n, p in enumerate(self.graph.nodes):
                p.params["col"] = cmap_vid[n]


        # PLOT Video
        if self.plot_vid:
            cv2.namedWindow("plot")


        # PLOT Network
        if self.plot_network and self.graph is not None:
            fig = plt.figure()
            ax = plt.gca()
            # ax.set_xlim(xlim[0], xlim[1])
            # ax.set_ylim(ylim[0], ylim[1])
            plt.ion()

        # plt.figure()
        # plt.scatter(np.arange(4), np.ones(4), color=cmap_)
        # plt.show()
        if self.out_name is not None:
            img_handle.open()
            rgb = img_handle.read_frame(0)
            img_handle.close()
            h, w, _ = rgb.shape
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(self.out_name, fourcc, 20.0, (w, h))

        img_handle.open()
        for t in range(self.graph.time_series_length):
            rgb = img_handle.read_frame(t)
            rgb_ = rgb.copy()

            # Plot info from graph
            if self.graph is not None:
                sc_x_ = list(map(int, sc_x[:, t]))
                sc_y_ = list(map(int, sc_y[:, t]))

                # Plot network
                if self.plot_network:
                    if self.network_scatter:
                        ax.scatter(sc_x_, sc_y_, color=cmap_network)

                    if self.network_lines:
                        for l in lines[t]:
                            ax.plot(l[0], l[1])

                    plt.pause(.1)
                self.plot_network = False

                # Plot video
                if self.plot_vid:
                    if self.vid_scatter:
                        for p in range(len(sc_x_)):
                            cv2.circle(rgb_, (sc_x_[p], sc_y_[p]), 1, tuple(cmap_vid[p]), 5)

                    if self.vid_lines:
                        for l in lines[t]:
                            cv2.line(rgb_, tuple(np.array(l[:, 0]).astype(int)), tuple(np.array(l[:, 1]).astype(int)),
                                     (255, 255, 255), 3)

            # Plot info from yolo
            if self.person_handle is not None and self.vid_bbox:
                for p in self.graph.nodes:
                    x_min, y_min, x_max, y_max = map(int, [p.params["xMin"][t], p.params["yMin"][t], p.params["xMax"][t], p.params["yMax"][t]])
                    NNHandler_person.plot(rgb_, (x_min, y_min, x_max, y_max), p.params["col"])

            # Plot info from openpose
            if self.openpose_handle is not None and self.vid_keypoints:
                NNHandler_openpose.plot(rgb_, self.openpose_handle.json_data[str(t)], self.openpose_handle.is_tracked)


            # Plot info from handshake
            if self.hs_handle is not None and self.vid_hbox:
                if str(t) in self.hs_handle.json_data:
                    NNHandler_handshake.plot(rgb_, self.hs_handle.json_data[str(t)], self.hs_handle.is_tracked)

            # Plot info from graph
            if self.graph is not None:
                if str(t) in self.graph.threatLevel:
                    cv2.putText(rgb_, str(self.graph.threatLevel[str(t)]), (100, 100), 0, 0.75, (255, 255, 255), 2)


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

            # save video
            if self.out_name is not None:
                out.write(rgb_)

            # display image with opencv or any operation you like
            cv2.imshow("plot", rgb_)

            k = cv2.waitKey(WAIT)

            if k & 0xff == ord('q'): break
            elif k & 0xff == ord('g') or WAIT != 0: self.plot_network = True

            # Plot network
            if self.plot_network:
                ax.clear()
                # ax.set_xlim(xlim[0], xlim[1])
                # ax.set_ylim(ylim[0], ylim[1])

        img_handle.close()

        if self.out_name is not None:
            out.release()
    # cap.release()

    # plt.show(block=True)


if __name__ == "__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument("--graph_file","-g",type=str,dest="graph_file",default='./data/vid-01-graph_handshake.json')
    parser.add_argument("--nnout_yolo","-y",type=str,dest="nnout_yolo",default='./data/labels/DEEE/yolo/cctv1-yolo.json')
    parser.add_argument("--nnout_handshake","-hs",type=str,dest="nnout_handshake",default='./data/labels/DEEE/handshake/cctv1.json')
    parser.add_argument("--video_file","-v",type=str,dest="video_file",default='./data/videos/DEEE/cctv1.mp4')
    # parser.add_argument("--graph_file","-g",type=str,dest="graph_file",default='./data/vid-01-graph_handshake.json')
    # parser.add_argument("--nnout_yolo","-y",type=str,dest="nnout_yolo",default='./data/vid-01-yolo.json')
    # parser.add_argument("--nnout_handshake","-hs",type=str,dest="nnout_handshake",default='./data/vid-01-handshake_track.json')
    # parser.add_argument("--video_file","-v",type=str,dest="video_file",default='./data/videos/seq18.avi')
    parser.add_argument("--nnout_openpose",'-p',type=str,dest="nnout_openpose",default='./data/vid-01-openpose_track.json')
    parser.add_argument("--config_file","-c",type=str,dest="config_file",default="args/visualizer-01.json")
    parser.add_argument("--output","-o",type=str,dest="output",default='./suren/temp/out.avi')
    parser.add_argument("--track", "-tr", type=bool, dest="track", default=True)

    args = parser.parse_args()
    print(args)

    # if None in [args.nnout_yolo, args.nnout_handshake, args.video_file, args.graph_file, args.track]:
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


    img_handle = NNHandler_image(format="avi", img_loc=args.video_file)
    img_handle.runForBatch()

    person_handler = NNHandler_person(args.nnout_yolo)
    person_handler.init_from_json()

    hs_handler = NNHandler_handshake(args.nnout_handshake, is_tracked=args.track)
    hs_handler.init_from_json()

    openpose_handler = NNHandler_openpose(openpose_file=args.nnout_openpose, is_tracked=args.track)
    openpose_handler.init_from_json()
    openpose_handler = None



    g = Graph()
    # g.init_from_json(args.graph_file)
    # g.run_gihan()

    if g.state["people"] < 2:
        person_handler.connectToGraph(g)
        person_handler.runForBatch()

    if g.state["handshake"] < 3:
        hs_handler.connectToGraph(g)
        hs_handler.runForBatch()

    # hs_handler = NNHandler_handshake('./data/vid-01-handshake.json', is_tracked=False)        # This is without DSORT tracker and avg
    # hs_handler = NNHandler_handshake('./data/vid-01-handshake_track.json', is_tracked=True)       # With DSORT and avg


    vis = Visualizer(graph=g, person=person_handler, handshake=hs_handler, img=img_handle, openpose=openpose_handler, out_name=None)  #args.output)

    # Call this to plot pyplot graph
    vis.init_network()
    # Call this to plot cv2 video
    vis.init_vid(vid_scatter=False, vid_lines=False)

    print("-------------------\nIf pyplot is visible and WAIT == 0, press 'g' to plot current graph\n-------------------")

    vis.plot(WAIT=0, show_cmap=False)

