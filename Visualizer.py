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

        # Info from img_handle
        self.start_time = 0
        self.end_time = None
        self.hw = None if self.img_handle is None else (img_handle.height, img_handle.width)

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

        if self.img_handle is not None:
            if self.img_handle.start_time is not None and self.img_handle.end_time is not None:
                self.start_time = self.img_handle.start_time
                self.end_time = self.img_handle.end_time
                self.time_series_length = self.end_time - self.start_time
            else:
                self.time_series_length = self.img_handle.time_series_length

        elif self.graph is not None:
            self.time_series_length = self.graph.time_series_length
        else:
            raise NotImplementedError

        # colour map
        if col_num is not None:
            self.cmap = self.get_cmap([col_num])
        elif self.graph is not None:
            self.cmap = self.get_cmap([self.graph.n_nodes])
            col_num = len(self.cmap)
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

            r, c = divmod(self.graph.n_nodes, len(cmap_plot))
            # print(cmap_plot.shape, r, c, self.graph.n_nodes)
            cmap_plot = np.append(np.tile(cmap_plot, (r, 1)), cmap_plot[:c, :], axis=0)
            # print(self.start_time, self.end_time, self.time_series_length)

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
            fig1, ax1 = plt.subplots(1)
            self.graph.image_init(ax1, xlim, ylim)

            # Figure for each metric
            fig2, ax2 = plt.subplots(2, 2)
            plt.subplots_adjust(wspace=.35, hspace=.35)
            fig4, ax4 = plt.subplots(1)
            self.graph.dimg_init(fig2, ax2, fig4, ax4)

            # Figure for threat level
            fig3, ax3 = plt.subplots(1)
            self.graph.threat_image_init(fig3, ax3)
            # fig3 = plt.figure()

        if self.img_handle is not None:
            self.img_handle.open(start_frame=start_time)

        for t in range(self.time_series_length):

            if self.img_handle is not None:
                rgb = self.img_handle.read_frame(t)
                rgb_ = rgb.copy()

            # ------------------------------- MAKE PLOT ----------------------------------

            # Plot network
            if self.make_plot:

                # Plot info from graph
                if self.graph is not None:
                    # sc_x_t = list(map(int, sc_x[:, t]))
                    # sc_y_t = list(map(int, sc_y[:, t]))
                    sc_x_t = sc_x[:, t]
                    sc_y_t = sc_y[:, t]

                    if self.plot_scatter:
                        ax1.scatter(sc_x_t, sc_y_t, color=cmap_plot)

                    if self.plot_lines:
                        for l in lines[t]:
                            ax1.plot(l[0], l[1])

                    if self.mark_ref:
                        for i in range(4):
                            p1x, p1y = self.graph.DEST[i]
                            p2x, p2y = self.graph.DEST[i-1]
                            ax1.plot([p1x, p2x], [-p1y, -p2y], 'k', linewidth=.5)


                    if self.plot_out_name is not None:
                        # Save graph
                        self.graph.image_save(fig1, ax1, xlim, ylim, self.plot_out_name, t, clear=True)
                        # Save DIMG
                        self.graph.dimg_save(fig2, ax2, fig4, ax4, self.plot_out_name, t)
                        # Save threat
                        self.graph.threat_image_save(fig3, ax3, self.plot_out_name, t)



                # @Suren : TODO : Learn ion properly -_-
                # self.plot_show = False

                # @suren : Scatters and lines are removed from video
                if self.make_vid:
                    if self.vid_scatter:
                        for p in range(len(sc_x_t)):
                            cv2.circle(rgb_, (sc_x_t[p], sc_y_t[p]), 1, tuple(cmap_vid[p%col_num]), 5)

                    if self.vid_lines:
                        for l in lines[t]:
                            cv2.line(rgb_, tuple(np.array(l[:, 0]).astype(int)), tuple(np.array(l[:, 1]).astype(int)),(255, 255, 255), 3)

            # ------------------------------- MAKE VIDEO ----------------------------------

            if self.make_vid:
                t_ = str(self.start_time + t)

                # Plot reference on video
                if self.mark_ref and self.graph is not None:
                    points = np.array([self.graph.REFERENCE_POINTS], dtype=np.int32)
                    cv2.polylines(rgb_, points, 1, 255)

                # Plot info of person
                if self.person_handle is not None and self.vid_bbox:
                    if t_ in self.person_handle.json_data:
                        NNHandler_person.plot(rgb_, self.person_handle.json_data[t_], cmap_vid, self.person_handle.is_tracked)
                        # TODO @suren : match colour in graph and vid


                if True:
                    for i in range(self.graph.n_nodes):
                        for j in range(i+1, self.graph.n_nodes):
                            if self.graph.pairG[t, i, j] > .99:
                                x1 = self.graph.nodes[i].params["X"][t]
                                y1 = self.graph.nodes[i].params["Y"][t]
                                x2 = self.graph.nodes[j].params["X"][t]
                                y2 = self.graph.nodes[j].params["Y"][t]

                                # print(t, i, j, "x1, y1, x2, y2 =", x1, y1, x2, y2)

                                cv2.line(rgb_, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 3)
                                cv2.circle(rgb_, (int(x1), int(y1)), 1, (0, 0, 0), 7)
                                cv2.circle(rgb_, (int(x2), int(y2)), 1, (0, 0, 0), 7)


                # Plot info from handshake
                if self.hs_handle is not None and self.vid_hbox:
                    if t_ in self.hs_handle.json_data:
                        NNHandler_handshake.plot(rgb_, self.hs_handle.json_data[t_], self.hs_handle.is_tracked)

                # Plot info from openpose
                if self.openpose_handle is not None and self.vid_keypoints:
                    NNHandler_openpose.plot(rgb_, self.openpose_handle.json_data[t_], self.openpose_handle.is_tracked)

                # Plot info from graph
                if self.graph is not None and self.graph.frameThreatLevel is not None:
                    font = 0 #cv2.FONT_HERSHEY_PLAIN
                    h, w, _ = rgb_.shape
                    offset_x, offset_y = int(.1*h), int(.9*h)
                    font_scale = w*.75 / float(720)
                    # print(font_scale, font, offset_x, offset_y)
                    cv2.putText(rgb_, "%.4f"%(self.graph.frameThreatLevel[t]), (offset_x, offset_y), font, font_scale, (0, 0, 0), 3)
                    cv2.putText(rgb_, "%.4f"%(self.graph.frameThreatLevel[t]), (offset_x, offset_y), font, font_scale, (255, 255, 255), 2)


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
        if directory is None: directory=self.plot_out_name

        out_dir = directory + "/merged/"
        if not os.path.exists(out_dir) : os.makedirs(out_dir)

        # Extension
        OUTPUT_FILE_TYPE="mp4"#"mp4" | "webm"

        # Output size : (w_out, h_out)
        h_out = 720

        h_pic = h_out // 2
        w2 = int(h_pic/0.75)

        try:
            img_temp = "{}fr-{:04d}.jpg".format(directory, 0)
            h, w, _ = cv2.imread(img_temp).shape
            w_pic = int(h_pic/float(h)*w)
        except FileNotFoundError:
            w_pic = int(h_pic/0.75)

        w_out = w2 + w_pic

        outVideoName = "{}merged.{}".format(out_dir, OUTPUT_FILE_TYPE)
        if OUTPUT_FILE_TYPE == "avi":
            mergedFourcc = cv2.VideoWriter_fourcc(*'XVID')
        elif OUTPUT_FILE_TYPE == "mp4":
            mergedFourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif OUTPUT_FILE_TYPE == "webm":
            mergedFourcc = cv2.VideoWriter_fourcc(*'VP90')
        else:
            raise NotImplementedError
        mergedVideoOut = cv2.VideoWriter(outVideoName, mergedFourcc, 20.0, (w_out, h_out))

        #This is a hardcoded function
        imgPefixes=["fr","G","dimg","T"]
        img_sizes = [(h_pic, w_pic), (h_out-h_pic, w_pic), (h_pic, w_out-w_pic), (h_out-h_pic, w_out-w_pic)]
        img_pad = [(0, 0), (h_pic, 0), (0, w_pic), (h_pic, w_pic)]

        fivePercentBlock= max(1, int(noFrames/20.0))
        print("0% of merging completed")        

        for t in range(noFrames):
            outImg = np.empty((h_out, w_out, 3), dtype=np.uint8)
            # outImg=np.zeros((newH,1,3),dtype=np.uint8)
            for i in range(4):
                imgName="{}{}-{:04d}.jpg".format(directory,imgPefixes[i],t)
                if args.debug:
                    print("Loading file ",imgName)

                thisImg=cv2.imread(imgName)
                out_size = img_sizes[i]
                thisImg=cv2.resize(thisImg,(out_size[1], out_size[0]))

                h_start = img_pad[i][0]
                h_end = h_start + out_size[0]
                w_start = img_pad[i][1]
                w_end = w_start + out_size[1]

                outImg[h_start:h_end, w_start:w_end] = thisImg
                # print("outimage shape",outImg.shape)

            cv2.imwrite("{}final-{:04d}.jpg".format(out_dir, t), outImg)
            mergedVideoOut.write(outImg)
            # print(newW,newH)

            if t%fivePercentBlock==0:
                print("{:.2f}% of merging completed".format((100*t)/noFrames))


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
    suren_mode = False
    start_time = 0
    end_time = 200
    col_num = 10

    if suren_mode:
        args.input = "./data/videos/DEEE/cctv1.mp4"
        args.person = "./data/labels/DEEE/yolo/cctv1-yolo.json"
        args.handshake = "./data/labels/DEEE/handshake/cctv1-hs.json"
        args.cam = "./data/camera-orientation/jsons/deee.json"
        args.graph = './data/temp/deee-cctv1.json'
        args.output = './data/output/DEEE/cctv1/'
        args.visualize = False
        # @gihan... change these
        start_time = 100
        end_time = 200
        col_num = 20
        # time_series_length = 500

    # Initiate image handler
    if args.input is not None:
        img_handle = NNHandler_image(format="avi", img_loc=args.input)
        img_handle.runForBatch(start_time, end_time)
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

    vis.plot(WAIT=20, col_num=col_num)

    if args.output is not None:
        vis.mergePhotos(noFrames=g.time_series_length)

    print("END of program")