from NNHandler import *
from yolo_handshake import YOLO_Handshake
from suren.util import get_iou
import  numpy as np


class NNHandler_handshake:
	def __int__(self, graph, yoloHandshakeOutput='yoloHandshakeOutput.txt'):
		print("Creating an Openpose handler")
		self.inputFileName=yoloHandshakeOutput
		self.graph = graph


	def runForBatch(self):
		
		# print("Getting frames from InputHandler")
		# frames=self.myInput.getFrameBlock(self.myId)
		
		# f=open(self.inputFileName,"r")
		#Get the handshake bounding boxes.


		#Use self.graph and find the two people using maximum intersection area


		# TODO SUREN ;-)

		yolo_handshake = YOLO_Handshake(self.inputFileName)

		assert yolo_handshake.TIME_SERIES_LENGTH == self.graph.TIME_SERIES_LENGTH, \
			"Both files (yolo and graph) must be of same length :/"

		# This is going to be inefficient:
		# Graph contains nodes which have time series info for separate nodes
		# YOLO output has timeseries info first and then info of each node for that time series

		for t in self.graph.TIME_SERIES_LENGTH:

			# First take all the detected nodes at time t
			node_t = []
			for node in self.graph.nodes:
				if node.params["detection"][t]:
					node_t.append([node.params["xMin"], node.params["xMax"], node.params["yMin"], node.params["yMax"]])

			# Next consider all YOLO bboxes at time t
			nbox = yolo_handshake.yolo_data[str(t)]["noBboxes"]
			for bbox in yolo_handshake.yolo_data[str(t)]["Bboxes"]:
				bb_yolo = [bbox["x1"], bbox["x2"], bbox["y1"], bbox["y2"]]

				iou = map(lambda x : get_iou(bb_yolo, x), node_t)

				max_ind = np.argmax(iou)




		# self.graph
		n=self.graph.getNode(0)
		n.addParam("handshake")
		for a in range(self.graph.timeSeriesLength):	# Why are there multiple time series lengths? Shouldn't this be global?
			n.setParam("handshake",{"person":None,"confidence":None})





		'''
		Put ur code here.
		'''

		print("Updated the graph")
		


	# def processBatch(self,fr):
	# 	print("NN in action")

