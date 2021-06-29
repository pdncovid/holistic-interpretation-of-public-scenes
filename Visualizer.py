import os
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
    import matplotlib
    matplotlib.use('Agg')

    # import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
except ImportError as e:
    print(e)


class Visualizer:


    @staticmethod
    def get_cmap(size : list):
        if len(size) == 1:
            n = size[0]
            cmap = plt.get_cmap('hsv')
            sample = np.linspace(0, 1, n+1)[:-1]
            colors = np.array([cmap(i) for i in sample])
            return colors
        elif len(size) == 2:
            n, window = size
            # cmap = plt.get_cmap('hsv')
            # window = 10
            cmap = plt.get_cmap('hsv')
            sample = np.linspace(0, 1, n+1)[:-1]
            colors = np.array([cmap(i) for i in sample])
            col_arr = np.ones((window, 4))
            col_arr[:, -1] = np.power(.8, np.arange(window))[::-1]
            arr1 = np.tile(colors, (window, 1, 1)).transpose((1, 0, 2))
            # print(colors.shape, arr1.shape)
            arr2 = np.tile(col_arr, (n, 1, 1))
            # print(col_arr.shape, arr2.shape)
            colors = arr1 * arr2
            return colors
        else:
            raise NotImplementedError



    def __init__(self, graph=None, person=None, handshake=None, img=None, openpose=None, debug=False):
        self.graph = graph
        self.person_handle = person
        self.hs_handle = handshake
        self.img_handle = img
        self.openpose_handle = openpose

        self.debug = debug
        self.time_series_length = None

        # Scatter plot components
        self.make_plot = False      # create plot (Everything below wont matter is this isn't set)
        self.plot_show = False      # show plot
        self.plot_out_name = None   # save plot
        self.plot_scatter = False
        self.plot_lines = False

        # Img/Video components
        self.make_vid = False       # create video frame (Everything below wont matter is this isn't set)
        self.vid_show = False      # show vid
        self.vid_out_name = None   # save vid
        self.img_out_name = None   # save vid as frames (only prefix here... )
        self.vid_bbox = False
        self.vid_hbox = False
        self.vid_scatter = False
        self.vid_lines = False
        self.vid_keypoints = False      # openpose

        # Common to plot and vid
        self.mark_ref =True

    def init_plot(self, plot_out : str = None, network_scatter=True, network_lines=True, network_show=False):
        assert self.graph is not None, "Graph cannot be empty while plotting"
        self.make_plot = True

        self.plot_out_name = plot_out
        self.plot_show = network_show
        self.plot_scatter = network_scatter
        self.plot_lines = network_lines

    def init_vid(self, vid_out : str = None, img_out : str = None,
                 vid_bbox=True, vid_hbox=True,  vid_scatter=False, vid_lines=False, vid_keypoints=True, vid_show = False):
        self.make_vid = True

        self.vid_out_name = vid_out
        self.img_out_name = img_out
        self.vid_show = vid_show
        self.vid_bbox = vid_bbox
        self.vid_hbox = vid_hbox
        self.vid_scatter = vid_scatter
        self.vid_lines = vid_lines
        self.vid_keypoints = vid_keypoints

    def plot(self, WAIT=20, col_num:int = None):

        if Graph.plot_import() is not None:
            eprint("Package not installed", Graph.plot_import())
            return

        assert self.graph is not None or self.img_handle is not None, "Cannot visualize anything if both Image handle and graph is None"

        if self.graph is not None:
            self.time_series_length = self.graph.time_series_length
        else:
            self.time_series_length = self.img_handle.time_series_length

        # colour map
        if col_num is not None:
            self.cmap = self.get_cmap([col_num])
        elif self.graph is not None:
            self.cmap = self.get_cmap([self.graph.n_nodes])
            # self.cmap = self.graph.get_cmap()
        else:
            raise NotImplementedError

        cmap_vid = np.array(self.cmap[:, :-1] * 255)[:, [2, 1, 0]]      # RGB and then to BGR
        cmap_plot = np.reshape(self.cmap, (-1, 4), order='C')        # RGB alpha
        # cmap_ = cv2.cvtColor(cmap_.reshape(1, -1, 3), cv2.COLOR_RGB2BGR).reshape(-1, 3)

        # Process and get all graph points till time t
        if self.graph is not None:

            # scatter x, y and lines
            sc_x, sc_y, lines = self.graph.get_plot_points()
            xlim, ylim = self.graph.get_plot_lim(sc_x, sc_y)
            # print(sc_x.shape, sc_y.shape, cmap.shape)

        # MAKE Video
        if self.make_vid:
            assert self.img_handle is not None, "Image handle cannot be None, if video is required"

            if self.vid_show:
                cv2.namedWindow("plot")

            # SAVE VIDEO
            if self.img_out_name is not None and not os.path.exists(self.img_out_name):
                os.makedirs(self.img_out_name)

            if self.vid_out_name is not None:
                if not os.path.exists(os.path.dirname(self.vid_out_name)):
                    os.makedirs(os.path.dirname(self.vid_out_name))

                self.img_handle.open()
                rgb = self.img_handle.read_frame()
                self.img_handle.close()
                h, w, _ = rgb.shape
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                vid_out = cv2.VideoWriter(self.vid_out_name, fourcc, 20.0, (w, h))

        # MAKE plot
        if self.make_plot:

            if self.plot_show: plt.ion()
            else: plt.ioff()

            if self.plot_out_name is not None and not os.path.exists(self.plot_out_name):
                os.makedirs(self.plot_out_name)

            # Figure for Floor points
            fig1, ax1 = plt.subplots()
            ax1.set_xlim(xlim[0], xlim[1])
            ax1.set_ylim(ylim[0], ylim[1])

            # Figure for each metric
            fig2, ax2 = plt.subplots(2, 2)
            plt.subplots_adjust(wspace=.35, hspace=.35)
            self.graph.gihan_init(fig2, ax2)

            # Figure for threat level
            fig3 = plt.figure()

        if self.img_handle is not None:
            self.img_handle.open()

        for t in range(self.time_series_length):

            if self.img_handle is not None:
                rgb = self.img_handle.read_frame(t)
                rgb_ = rgb.copy()

            # ------------------------------- MAKE PLOT ----------------------------------

            # Plot network
            if self.make_plot:

                # Plot info from graph
                if self.graph is not None:
                    sc_x_t = list(map(int, sc_x[:, t]))
                    sc_y_t = list(map(int, sc_y[:, t]))

                    if self.plot_scatter:
                        ax1.scatter(sc_x_t, sc_y_t, color=cmap_plot)

                    if self.plot_lines:
                        for l in lines[t]:
                            ax1.plot(l[0], l[1])

                    if self.mark_ref:
                        for i in range(4):
                            p1, p2 = self.graph.DEST[i], self.graph.DEST[i - 1]
                            ax1.plot(p1, p2, 'k', linewidth=.5)


                    if self.plot_out_name is not None:
                        fig1.savefig("{}G-{:04d}.jpg".format(self.plot_out_name, t))
                        ax1.clear()
                        ax1.set_xlim(xlim[0], xlim[1])
                        ax1.set_ylim(ylim[0], ylim[1])

                        self.graph.gihan_images(fig2, ax2, self.plot_out_name, t)
                        self.graph.threat_image(fig3, self.plot_out_name, t)



                # @Suren : TODO : Learn ion properly -_-
                # self.plot_show = False

                # @suren : Scatters and lines are removed from video
                if self.make_vid:
                    if self.vid_scatter:
                        for p in range(len(sc_x_t)):
                            cv2.circle(rgb_, (sc_x_t[p], sc_y_t[p]), 1, tuple(cmap_vid[p]), 5)

                    if self.vid_lines:
                        for l in lines[t]:
                            cv2.line(rgb_, tuple(np.array(l[:, 0]).astype(int)), tuple(np.array(l[:, 1]).astype(int)),(255, 255, 255), 3)

            # ------------------------------- MAKE VIDEO ----------------------------------

            if self.make_vid:

                # Plot reference on video
                if self.mark_ref and self.graph is not None:
                    points = np.array([self.graph.REFERENCE_POINTS], dtype=np.int32)
                    cv2.polylines(rgb_, points, 1, 255)

                # Plot info from yolo
                if self.person_handle is not None and self.vid_bbox:
                    if str(t) in self.person_handle.json_data:
                        NNHandler_person.plot(rgb_, self.person_handle.json_data[str(t)], cmap_vid, self.person_handle.is_tracked)
                        # TODO @suren : match colour in graph and vid

                # Plot info from handshake
                if self.hs_handle is not None and self.vid_hbox:
                    if str(t) in self.hs_handle.json_data:
                        NNHandler_handshake.plot(rgb_, self.hs_handle.json_data[str(t)], self.hs_handle.is_tracked)

                # Plot info from openpose
                if self.openpose_handle is not None and self.vid_keypoints:
                    NNHandler_openpose.plot(rgb_, self.openpose_handle.json_data[str(t)], self.openpose_handle.is_tracked)

                # Plot info from graph
                if self.graph is not None and self.graph.frameThreatLevel is not None:
                    cv2.putText(rgb_, "%.4f"%(self.graph.frameThreatLevel[t]), (100, 100), 0, 0.75, (255, 255, 255), 2)

                # save video
                if self.vid_out_name is not None:
                    vid_out.write(rgb_)

                if self.plot_out_name is not None:
                    if self.debug: print("{}fr-{:04d}.jpg".format(self.img_out_name, t))
                    cv2.imwrite("{}fr-{:04d}.jpg".format(self.img_out_name, t), rgb_)

            # ------------------------------- SHOW VIDEO / PLOT ----------------------------------

            # display image with opencv or any operation you like
            if self.make_vid and self.vid_show:
                cv2.imshow("plot", rgb_)

                k = cv2.waitKey(WAIT)

                if k & 0xff == ord('q'): break
                elif k & 0xff == ord('g') or WAIT != 0: pass # self.network_show = True


            if self.make_plot and self.plot_show:
                ax1.clear()
                ax1.set_xlim(xlim[0], xlim[1])
                ax1.set_ylim(ylim[0], ylim[1])

            if (t + 1) % 20 == 0:
                progress(t + 1, self.time_series_length, "drawing graph")

        if self.img_handle is not None:
            self.img_handle.close()

        if self.vid_out_name is not None:
            vid_out.release()
    # cap.release()

    # plt.show(block=True)


    def mergePhotos(self,directory=None,noFrames=100):
        OUTPUT_FILE_TYPE="mp4"#"mp4" | "webm"


        mergedVideoOut="TempVariable"
        newH=500
        if directory==None:
            directory=self.plot_out_name

        #This is a hardcoded function
        imgPefixes=["fr","G","dimg","T"]

        fivePercentBlock= min(1, int(noFrames/20.0))
        print("0% of merging completed")        

        for t in range(noFrames):
            outImg=np.zeros((newH,1,3),dtype=np.uint8)
            for i in range(len(imgPefixes)):
                imgName="{}{}-{:04d}.jpg".format(directory,imgPefixes[i],t)
                if args.debug:
                    print("Loading file ",imgName)
                thisImg=cv2.imread(imgName)
                H=thisImg.shape[0]
                W=thisImg.shape[1]

                newW=int((newH/(1.0*H))*W)
                thisImg=cv2.resize(thisImg,(newW,newH))
                outImg=np.concatenate((outImg,thisImg),axis=1)
                # print("outimage shape",outImg.shape)

            cv2.imwrite("{}final-{:04d}.jpg".format(self.plot_out_name, t), outImg)

            if t==0:
                if OUTPUT_FILE_TYPE=="mp4":
                    outVideoName="{}merged.mp4".format(directory)
                    mergedFourcc = cv2.VideoWriter_fourcc(*'XVID')
                elif OUTPUT_FILE_TYPE=="webm":
                    outVideoName="{}merged.webm".format(directory)
                    mergedFourcc = cv2.VideoWriter_fourcc(*'VP90')
                mergedVideoOut = cv2.VideoWriter(outVideoName, mergedFourcc, 20.0, (int(outImg.shape[1]), int(outImg.shape[0])))
            mergedVideoOut.write(outImg)
            # print(newW,newH)

            if t%fivePercentBlock==0:
                print((5.0*t)/fivePercentBlock,"% of merging completed")


        mergedVideoOut.release()
        print("100% of merging completed")

if __name__ == "__main__":

    parser=argparse.ArgumentParser()

    # IGNORE THIS
    # parser.add_argument("--nnout_openpose",'-p',type=str,dest="nnout_openpose",default='./data/vid-01-openpose_track.json')

    parser.add_argument("--input","-i", type=str, default='./data/videos/seq18.avi') # Change this : input fil
    parser.add_argument("--output","-o", type=str, default='./data/output/seq18-temp/') # Change this : output dir
    parser.add_argument("--person","-p", type=str, default='./data/labels/seq18/seq18-person.json') # Change this : person
    parser.add_argument("--handshake","--hs", type=str, default='./data/labels/seq18/seq18-handshake.json') # Change this : handshake
    parser.add_argument("--cam", "-c", type=str, default="./data/camera-orientation/jsons/uti.json") # Change this: camfile
    parser.add_argument("--graph","-g", type=str, default='./data/output/seq18/seq18-graph-temp.json') # Change this : INCOMPLETE (Make sure this isn't None)

    parser.add_argument("--visualize","-v", action="store_true", help="Visualize the video output") # Change this

    parser.add_argument("--overwrite_graph","-owg", type=bool, default=False) # Change this : INCOMPLETE
    parser.add_argument("--track", "-tr", type=bool, dest="track", default=True)
    parser.add_argument("--debug", "-db", type=bool, dest="debug", default=False)

    args = parser.parse_args()
    print(args)

    # args.visualize = True
    # args.output = None
    # args.graph = None
    # time_series_length = 500
    suren_mode = True
    start_time = 0
    end_time = None

    if suren_mode:
        args.input = "./data/videos/TownCentreXVID.mp4"
        args.person = "./data/labels/TownCentre/person.json"
        args.handshake = "./data/labels/TownCentre/person.json"
        args.cam = "./data/camera-orientation/jsons/oxford.json"
        args.graph = './data/temp/oxford-graph.json'
        args.output = './data/output/oxford/'
        args.visualize = False
        # @gihan... change these
        start_time = 100
        end_time = 500
        # time_series_length = 500

    # Initiate image handler
    if args.input is not None:
        img_handle = NNHandler_image(format="avi", img_loc=args.input)
        img_handle.runForBatch()
    else:
        img_handle = None

    # Person handler
    if args.person is not None:
        person_handler = NNHandler_person(args.person, is_tracked=args.track)
        if os.path.exists(args.person):
            person_handler.init_from_json()
        else:
            person_handler.create_yolo(img_handle)
            person_handler.save_json()
    else:
        person_handler = None

    # HS handler
    if args.handshake is not None:
        hs_handler = NNHandler_handshake(args.handshake, is_tracked=args.track)
        if os.path.exists(args.handshake):
            hs_handler.init_from_json()
        else:
            hs_handler.create_yolo(img_handle)
            hs_handler.save_json()
    else:
        hs_handler = None


    # openpose_handler = NNHandler_openpose(openpose_file=args.nnout_openpose,  is_tracked=args.track)
    # openpose_handler.init_from_json()
    openpose_handler = None

    if args.graph is not None:
        g = Graph()
        g.getCameraInfoFromJson(args.cam)

        if os.path.exists(args.graph):
            g.init_from_json(args.graph)

        print("State = ", g.state)

        if g.state["people"] < 2:
            person_handler.connectToGraph(g)
            person_handler.runForBatch(start_time, end_time)

        if g.state["handshake"] < 2:
            hs_handler.connectToGraph(g)
            hs_handler.runForBatch(start_time, end_time)

        if g.state["floor"] < 7:
            g.generateFloorMap()

        if g.state["cluster"] < 1:
            g.findClusters()

        if g.state["threat"] < 1:
            g.calculateThreatLevel()

        if args.overwrite_graph:
            g.saveToFile(args.graph)
    else:
        g = None

    vis = Visualizer(graph=g, person=person_handler, handshake=hs_handler, img=img_handle, openpose=openpose_handler)

    if args.output is not None:
        plot_loc = args.output + "/plot/"
        vid_loc = (args.output + "/out.avi").replace("\\", "/").replace("//", "/")
    else:
        plot_loc = vid_loc = None

    # Call this to plot pyplot graph
    if args.output is not None:
        vis.init_plot(plot_out=plot_loc)

    # Call this to plot cv2 video
    if args.output is not None or args.visualize is not None:
        vis.init_vid(vid_out= vid_loc, img_out=plot_loc, vid_show=args.visualize)

    print("-----------------\nIf pyplot is visible and WAIT == 0, press 'g' to plot current graph\n-----------------")

    vis.plot(WAIT=20)

    if args.output is not None:
        vis.mergePhotos(noFrames=g.time_series_length)

    print("END of program")