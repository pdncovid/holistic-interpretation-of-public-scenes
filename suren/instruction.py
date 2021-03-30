import argparse

from Graph import Graph
from NNHandler_handshake import NNHandler_handshake
from NNHandler_image import NNHandler_image
from NNHandler_yolo import NNHandler_yolo
from Visualizer import Visualizer

from suren.util import eprint, progress, Json
try:
    # import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
except ImportError as e:
    print(e)



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

    img_handle = NNHandler_image(format="avi", img_loc=args.video_file)
    img_handle.runForBatch()

    yolo_handler = NNHandler_yolo(is_tracked=True)
    yolo_handler.create_tracker(img_handle)
    yolo_handler.save_json('xx1.json')

    yolo_handler.connectToGraph(g)
    yolo_handler.runForBatch()

    hs_handler = NNHandler_handshake(is_tracked=True)
    hs_handler.create_tracker(img_handle)
    hs_handler.save_json('xx2.json')

    hs_handler.connectToGraph(g)
    hs_handler.runForBatch()

    vis = Visualizer(graph= g, yolo=yolo_handler, handshake=hs_handler, img=img_handle)
    vis.plot(WAIT=0)

