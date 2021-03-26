from NNHandler import *
# from yolo_handshake import YOLO_Handshake
from NNHandler_image import NNHandler_image
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

		with open(handshake_file) as json_file:
			data = json.load(json_file)

		self.json_data = data
		self.time_series_length = self.json_data["frames"]

		return self.json_data

	def update_handshake(self, handshake_file=None):
		# Use self.graph and find the two people using maximum intersection area
		# TODO SUREN ;-)

		handshake_file = self.handshake_file if handshake_file is None else handshake_file

		# Read json and return data in self.yolo_data
		handshake_data = self.init_from_json(handshake_file)

		assert self.time_series_length == self.graph.time_series_length, \
			"Both files (yolo and graph) must be of same length :/ (%d, %d)"%(self.time_series_length, self.graph.time_series_length)

		# This is going to be inefficient:
		# Graph contains nodes which have time series info for separate nodes
		# YOLO output has timeseries info first and then info of each node for that time series

		handshake_frames = list(map(int, list(handshake_data.keys())[1:]))	# write in a better way
		print(handshake_frames)


		for t in handshake_frames:

			# First take all the detected nodes at time t
			node_t = []
			node_ind = []
			for ind, node in enumerate(self.graph.nodes):
				if node.params["detection"][t]:
					node_t.append([node.params["xMin"][t], node.params["xMax"][t], node.params["yMin"][t], node.params["yMax"][t]])
					node_ind.append(ind)
			# Next consider all YOLO bboxes at time t
			
			nbox = handshake_data[str(t)]["No of boxes"]

			for bbox in handshake_data[str(t)]["bboxes"]:
				bb_hs = [bbox["x1"], bbox["x2"], bbox["y1"], bbox["y2"]]

				conf = bbox["conf"]

				# iou between bb_hs and bb_person (node_t)

				iou = []
				for i in range(len(node_t)):
					try:
						iou.append(get_iou(bb_hs, node_t[i], mode=1))
					except Exception as e:
						print(e)
						print(t, i)
						input("Enter something")

				# iou = list(map(lambda x: get_iou(bb_hs, x, mode=1), node_t))

				# get 2 max values
				ind1, ind2 = np.argpartition(iou, -2)[-2:]

				p1, p2 = node_ind[ind1], node_ind[ind2]

				# p1, p2 = node_ind[np.array([ind1, ind2])]

				self.graph.nodes[p1].params["handshake"][t] = {"person": p2, "confidence": conf}
				self.graph.nodes[p2].params["handshake"][t] = {"person": p1, "confidence": conf}

		print("Updated the graph")

	# def processBatch(self,fr):
	# 	print("NN in action")

	def runForBatch(self):

		self.update_handshake()
		




if __name__ == "__main__":
	g = Graph()
	g.init_from_json('./data/vid-01-graph.json')

	print("Created graph with nodes = %d for frames = %d. Param example:" % (g.n_nodes, g.time_series_length))
	print(g.nodes[0].params)

	# g.plot()
	print()

	nn_handshake = NNHandler_handshake(g, './data/vid-01-handshake.json')
	nn_handshake.init_from_json()
	nn_handshake.update_handshake()

	print("Initiated handshake for %s nodes. Param example:"%g.n_nodes)
	for param in g.nodes[0].params:
		print(param, g.nodes[0].params[param])

	g.saveToFile("./data/vid-01-graph_handshake.json")






