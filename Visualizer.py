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
    def __init__(self, graph=None, person=None, handshake=None, img=None, openpose=None):
        self.graph = graph
        self.person_handle = person
        self.hs_handle = handshake
        self.img_handle = img
        self.openpose_handle = openpose

        # Scatter plot components
        self.plot_network = False
        self.network_show = False
        self.network_scatter = False
        self.network_lines = False
        self.plot_out = None

        # Img/Video components
        self.plot_vid = False
        self.vid_bbox = False
        self.vid_hbox = False
        self.vid_scatter = False
        self.vid_lines = False
        self.vid_keypoints = False
        self.vid_out = None

    def init_network(self, plot_out : str = None, network_scatter=True, network_lines=True, network_show=False):
        assert self.graph is not  None, "Graph cannot be empty while plotting"

        self.plot_out  = plot_out
        self.plot_network = True
        self.network_show = network_show
        self.network_scatter = network_scatter
        self.network_lines = network_lines

    def init_vid(self, vid_out : str = None, vid_bbox=True, vid_hbox=True, vid_scatter=True, vid_lines=True, vid_keypoints=True):
        self.vid_out = vid_out
        self.plot_vid = True
        self.vid_bbox = vid_bbox
        self.vid_hbox = vid_hbox
        self.vid_scatter = vid_scatter
        self.vid_lines = vid_lines
        self.vid_keypoints = vid_keypoints

    def plot(self, WAIT=20, show_cmap=True):
        self.show_gui=False
        self.save_video_frames=True
        self.plot_vid=True#Gihan added this because he didn't understand the code


        if Graph.plot_import() is not None:
            eprint("Package not installed", Graph.plot_import())
            return

        # Process and get all graph points till time t
        if self.graph is not None:
            # colour map
            cmap = self.graph.get_cmap(show=show_cmap)

            # scatter x, y and lines
            sc_x, sc_y, lines = self.graph.get_plot_points()

            xlim, ylim = self.graph.get_plot_lim(sc_x, sc_y)

            # print(sc_x.shape, sc_y.shape, cmap.shape)

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

            # SAVE VIDEO
            if self.vid_out is not None:
                img_handle.open()
                rgb = img_handle.read_frame(0)
                img_handle.close()
                h, w, _ = rgb.shape
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                vid_out = cv2.VideoWriter(self.vid_out, fourcc, 20.0, (w, h))


        # PLOT Network
        if self.plot_network:

            if self.network_show:
                plt.ion()
            else:
                plt.ioff()

            if self.plot_out is not None and not os.path.exists(self.plot_out):
                os.makedirs(self.plot_out)

            # Figure for Floor points
            fig1, ax1 = plt.subplots()
            ax1.set_xlim(xlim[0], xlim[1])
            ax1.set_ylim(ylim[0], ylim[1])

            # Figure for each metric
            fig2, ax2 = plt.subplots(2, 2)
            plt.subplots_adjust(wspace=.35, hspace=.35)
            g.gihan_init(fig2, ax2)

            # Figure for threat level
            fig3 = plt.figure()



        # plt.figure()
        # plt.scatter(np.arange(4), np.ones(4), color=cmap_)
        # plt.show()


        img_handle.open()
        for t in range(self.graph.time_series_length):
            rgb = img_handle.read_frame(t)
            rgb_ = rgb.copy()

            # Plot info from graph
            if self.graph is not None:
                sc_x_t = list(map(int, sc_x[:, t]))
                sc_y_t = list(map(int, sc_y[:, t]))

                # Plot network
                if self.plot_network:
                    if self.network_scatter:
                        ax1.scatter(sc_x_t, sc_y_t, color=cmap_network)

                    if self.network_lines:
                        for l in lines[t]:
                            ax1.plot(l[0], l[1])

                    if self.show_gui:
                        plt.pause(.1)

                # @Suren : TODO : Learn ion properly -_-
                self.network_show = False

                # Plot video
                if self.plot_vid:
                    if self.vid_scatter:
                        for p in range(len(sc_x_t)):
                            cv2.circle(rgb_, (sc_x_t[p], sc_y_t[p]), 1, tuple(cmap_vid[p]), 5)

                    if self.vid_lines:
                        for l in lines[t]:
                            cv2.line(rgb_, tuple(np.array(l[:, 0]).astype(int)), tuple(np.array(l[:, 1]).astype(int)),
                                     (255, 255, 255), 3)

            # print("DEBUG",self.plot_vid)
            if self.plot_vid:
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
                if self.graph is not None and self.graph.threatLevel is not None:
                    if str(t) in self.graph.threatLevel:
                        cv2.putText(rgb_, str(self.graph.threatLevel[str(t)]), (100, 100), 0, 0.75, (255, 255, 255), 2)

                # save video
                if self.vid_out is not None:
                    vid_out.write(rgb_)



                # display image with opencv or any operation you like
                if self.show_gui:
                    cv2.imshow("plot", rgb_)

                    k = cv2.waitKey(WAIT)

                    if k & 0xff == ord('q'): break
                    elif k & 0xff == ord('g') or WAIT != 0: pass # self.network_show = True

                
                if self.save_video_frames:
                    if args.debug:
                        print("{}fr-{:04d}.jpg".format(self.plot_out,t))
                    cv2.imwrite("{}fr-{:04d}.jpg".format(self.plot_out,t), rgb_)                    
            '''
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
            '''



            if self.plot_network:
                if self.plot_out is not None:
                    fig1.savefig("{}G-{:04d}.jpg".format(self.plot_out, t))
                    ax1.clear()
                    ax1.set_xlim(xlim[0], xlim[1])
                    ax1.set_ylim(ylim[0], ylim[1])

                    g.gihan_images(fig2, ax2, self.plot_out, t)
                    g.threat_image(fig3, self.plot_out, t)

                elif self.network_show:
                    ax1.clear()
                    ax1.set_xlim(xlim[0], xlim[1])
                    ax1.set_ylim(ylim[0], ylim[1])

            if (t + 1) % 20 == 0:
                progress(t + 1, self.graph.time_series_length, "drawing graph")

        img_handle.close()

        if self.vid_out is not None:
            vid_out.release()
    # cap.release()

    # plt.show(block=True)


    def mergePhotos(self,directory=None):
        mergedVideoOut=None
        newH=500
        if directory==None:
            directory=self.plot_out

        #This is a hardcoded function
        imgPefixes=["fr","G","dimg","T"]
        # imgPefixes=["fr","G","dimg","T"]
        for t in range(900):
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

            cv2.imwrite("{}final-{:04d}.jpg".format(self.plot_out,t), outImg)

            if t==0:
                outVideoName="{}merged.mp4".format(directory)
                mergedFourcc = cv2.VideoWriter_fourcc(*'XVID')
                mergedVideoOut = cv2.VideoWriter(outVideoName,\
                 mergedFourcc, 20.0, (int(outImg.shape[1]), int(outImg.shape[0])))
            mergedVideoOut.write(outImg)
            # print(newW,newH)


        mergedVideoOut.release()


if __name__ == "__main__":

    parser=argparse.ArgumentParser()
    # parser.add_argument("--graph_file","-g",type=str,dest="graph_file",default='./data/vid-01-graph_handshake.json')
    # parser.add_argument("--nnout_yolo","-y",type=str,dest="nnout_yolo",default='./data/labels/DEEE/yolo/cctv1-yolo.json')
    # parser.add_argument("--nnout_handshake","-hs",type=str,dest="nnout_handshake",default='./data/labels/DEEE/handshake/cctv1.json')
    # parser.add_argument("--video_file","-v",type=str,dest="video_file",default='./data/videos/DEEE/cctv1.mp4')
    parser.add_argument("--graph_file","-g",type=str,dest="graph_file",default='./data/vid-01-graph_handshake.json')
    parser.add_argument("--nnout_yolo","-y",type=str,dest="nnout_yolo",default='./data/vid-01-yolo.json')
    parser.add_argument("--nnout_handshake","-hs",type=str,dest="nnout_handshake",default='./data/vid-01-handshake_track.json')
    parser.add_argument("--video_file","-v",type=str,dest="video_file",default='./data/videos/seq18.avi')
    parser.add_argument("--nnout_openpose",'-p',type=str,dest="nnout_openpose",default='./data/vid-01-openpose_track.json')
    parser.add_argument("--config_file","-c",type=str,dest="config_file",default="args/visualizer-01.json")
    parser.add_argument("--output","-o",type=str,dest="output",default='./suren/temp/out.avi')
    parser.add_argument("--track", "-tr", type=bool, dest="track", default=True)
    parser.add_argument("--debug", "-db", type=bool, dest="debug", default=False)

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
    g.getCameraInfoFromJson("./data/camera-orientation/jsons/uti.json")
    # g.init_from_json(args.graph_file)
    # g.run_gihan()

    if g.state["people"] < 2:
        person_handler.connectToGraph(g)
        person_handler.runForBatch()

    if g.state["handshake"] < 3:
        hs_handler.connectToGraph(g)
        hs_handler.runForBatch()

    if g.state["floor"] < 2:
        g.generateFloorMap()

    if g.state["cluster"] < 1:
        g.findClusters()

    g.calculateThreatLevel()

    # hs_handler = NNHandler_handshake('./data/vid-01-handshake.json', is_tracked=False)        # This is without DSORT tracker and avg
    # hs_handler = NNHandler_handshake('./data/vid-01-handshake_track.json', is_tracked=True)       # With DSORT and avg


    vis = Visualizer(graph=g, person=person_handler, handshake=hs_handler, img=img_handle, openpose=openpose_handler)  #args.output)
    # Call this to plot pyplot graph
    vis.init_network(plot_out="./data/output/vid-01/plot/")
    # Call this to plot cv2 video
    # vis.init_vid(vid_out="./data/output/vid-01/out.mp4", vid_scatter=False, vid_lines=False)

    print("-------------------\nIf pyplot is visible and WAIT == 0, press 'g' to plot current graph\n-------------------")

    vis.plot(WAIT=20, show_cmap=False)

    vis.mergePhotos()
    

    print("END of program")