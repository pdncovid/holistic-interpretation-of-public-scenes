from NNHandler import *
from yolo_handshake import YOLO_Handshake
from suren.util import get_iou
import  numpy as np


class NNHandler_handshake(NNHandler):
	def __int__(self, graph, handshake_file):
		super().__init__()

		print("Creating a Handshake handler")
		self.graph = graph
		self.handshake_file = handshake_file


	def runForBatch(self):
		
		# print("Getting frames from InputHandler")
		# frames=self.myInput.getFrameBlock(self.myId)
		
		# f=open(self.inputFileName,"r")
		#Get the handshake bounding boxes.


		#Use self.graph and find the two people using maximum intersection area


		# TODO SUREN ;-)

		yolo_handshake = YOLO_Handshake(self.handshake_file)

		assert yolo_handshake.TIME_SERIES_LENGTH == self.graph.TIME_SERIES_LENGTH, \
			"Both files (yolo and graph) must be of same length :/"

		# This is going to be inefficient:
		# Graph contains nodes which have time series info for separate nodes
		# YOLO output has timeseries info first and then info of each node for that time series

		for t in self.graph.TIME_SERIES_LENGTH:

			# First take all the detected nodes at time t
			node_t = []
			node_ind = []
			for ind, node in enumerate(self.graph.nodes):
				if node.params["detection"][t]:
					node_t.append([node.params["xMin"], node.params["xMax"], node.params["yMin"], node.params["yMax"]])
					node_ind.append(ind)


			# Next consider all YOLO bboxes at time t
			nbox = yolo_handshake.yolo_data[str(t)]["noBboxes"]

			for bbox in yolo_handshake.yolo_data[str(t)]["Bboxes"]:
				bb_yolo = [bbox["x1"], bbox["x2"], bbox["y1"], bbox["y2"]]

				conf = 0	# @jameel, write the confidence val too.

				# iou between bb_hs and bb_yolo
				iou = map(lambda x : get_iou(bb_yolo, x), list(node_t.values()))

				# get 2 max values
				ind1, ind2 = np.argpartition(iou, -2)[-2:]

				p1, p2 = np.array(node_ind)[ind1, ind2]

				self.graph[p1].params["handshake"][t] = {"person": p2, "confidence": conf}
				self.graph[p2].params["handshake"][t] = {"person": p1, "confidence": conf}


		print("Updated the graph")
		


	# def processBatch(self,fr):
	# 	print("NN in action")

