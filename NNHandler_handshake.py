from NNHandler import *
# from yolo_handshake import YOLO_Handshake
from suren.util import get_iou, Json
import  numpy as np

from Graph import Graph


class NNHandler_handshake(NNHandler):

	def __init__(self, graph, handshake_file):

		print("Creating a Handshake handler")
		self.graph = graph
		self.handshake_file = handshake_file

	def init_from_json(self, handshake_file=None):
		handshake_file = self.handshake_file if handshake_file is None else handshake_file

		js = Json(handshake_file)

		self.json_data = js.read()
		self.time_series_length = self.json_data["frames"]

		return self.json_data

	def update_handshake(self, handshake_file=None):
		# Use self.graph and find the two people using maximum intersection area
		# TODO SUREN ;-)

		handshake_file = self.handshake_file if handshake_file is None else handshake_file

		# Read json and return data in self.yolo_data
		handshake_data = self.init_from_json(handshake_file)

		assert self.time_series_length == self.graph.time_series_length, \
			"Both files (yolo and graph) must be of same length :/"

		# This is going to be inefficient:
		# Graph contains nodes which have time series info for separate nodes
		# YOLO output has timeseries info first and then info of each node for that time series

		for t in range(self.graph.time_series_length):

			# First take all the detected nodes at time t
			node_t = []
			node_ind = []
			for ind, node in enumerate(self.graph.nodes):
				if node.params["detection"][t]:
					node_t.append([node.params["xMin"], node.params["xMax"], node.params["yMin"], node.params["yMax"]])
					node_ind.append(ind)
			# Next consider all YOLO bboxes at time t
			
			try:
				nbox = handshake_data[str(t)]["noBboxes"]
				print("AAA",t)

				for bbox in handshake_data[str(t)]["Bboxes"]:
					bb_hs = [bbox["x1"], bbox["x2"], bbox["y1"], bbox["y2"]]

					conf = bbox["conf"]

					# iou between bb_hs and bb_person (node_t)
					iou = map(lambda x: get_iou(bb_hs, x, mode=1), node_t)

					# get 2 max values
					ind1, ind2 = np.argpartition(iou, -2)[-2:]

					p1, p2 = np.array(node_ind)[ind1, ind2]

					self.graph[p1].params["handshake"][t] = {"person": p2, "confidence": conf}
					self.graph[p2].params["handshake"][t] = {"person": p1, "confidence": conf}
			except:
				_=0

		print("Updated the graph")

	# def processBatch(self,fr):
	# 	print("NN in action")

	def runForBatch(self):

		self.update_handshake()
		




if __name__ == "__main__":
	g = Graph()
	g.init_from_json('./nn-outputs/sample-YOLO-bbox.json')

	nn_handshake = NNHandler_handshake(g, './nn-outputs/sample-handshake-output.json')
	nn_handshake.init_from_json()

	for p in g.nodes:
		print(p.params)

