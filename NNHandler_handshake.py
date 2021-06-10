import numpy as np
import json
import os, sys
# from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse

from NNHandler_yolo import NNHandler_yolo
from NNHandler_image import NNHandler_image, cv2
from Graph import Graph

from suren.util import get_iou, Json, eprint

# This is only needed if running YOLO / deepsort
# Not needed if the values are loaded from file
try:
	import tensorflow as tf
	from tensorflow.python.saved_model import tag_constants

	sys.path.append(os.path.relpath('./suren/temp/yolov4-deepsort-master'))

	from deep_sort import preprocessing, nn_matching
	from deep_sort.detection import Detection
	from deep_sort.tracker import Tracker
	from tools import generate_detections as gdet
	import core.utils as utils
	# from core.yolov4 import filter_boxes
	from tensorflow.python.saved_model import tag_constants

	from core.config import cfg
except Exception as e:
	print(e)
	print("If YOLO isn't required, ignore this")


# def import_tracker(name="deepsort"):
# 	if name == "deepsort":
# 		try:
#
# 			from deep_sort.tracker import Tracker, nn_matching
# 			from deep_sort.detection import Detection
# 			from deep_sort.tracker import Tracker
# 			from tools import generate_detections as gdet
# 			return True
# 		except:
# 			eprint("Deepsort not installed.")
# 			return False
#
# 	else:
# 		raise NotImplementedError


class NNHandler_handshake(NNHandler_yolo):
	yolo_dir = "./suren/temp/yolov4-deepsort-master/"
	model_filename = yolo_dir + 'model_data/mars-small128.pb'
	weigths_filename = yolo_dir + '/checkpoints/yolov4-fullshake_best'

	class_names =  ["Handshake"]

	# Definition of the parameters
	max_cosine_distance = 0.4
	nn_budget = None
	nms_max_overlap = 1.0

	iou_thresh = .45
	score_thresh = .5
	input_size = 416


	@staticmethod
	def plot(img, points, is_tracked):
		if is_tracked:
			bb_dic = points
		else:
			bb_dic = points["bboxes"]

		for bbox in bb_dic:
			x_min, x_max, y_min, y_max = map(int, [bbox["x1"], bbox["x2"], bbox["y1"], bbox["y2"]])
			cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)

	def __init__(self, handshake_file=None, is_tracked=True, vis=True, verbose=True, debug=False):

		super().__init__(json_file=handshake_file, is_tracked=is_tracked, vis=vis, verbose=verbose, debug=debug)
		print("\t[*] Handshake detector")

	def update_handshake(self):
		# Use self.graph and find the two people using maximum intersection area

		graph = self.graph
		handshake_data = self.json_data

		# assert self.time_series_length == self.graph.time_series_length, \
		# 	"Both files (yolo and graph) must be of same length :/ (%d, %d)" % (
		# 	self.time_series_length, self.graph.time_series_length)

		# This is going to be inefficient:
		# Graph contains nodes which have time series info for separate nodes
		# YOLO output has timeseries info first and then info of each node for that time series

		handshake_frames = list(map(int, list(handshake_data.keys())))  # write in a better way
		# print(handshake_frames)

		if self.is_tracked:
			shakes = defaultdict(dict)

			for t in handshake_frames:
				# First take all the detected nodes at time t
				node_t = []
				node_ind = []
				for ind, node in enumerate(graph.nodes):
					if node.params["detection"][t]:
						node_t.append([node.params["xMin"][t], node.params["yMin"][t], node.params["xMax"][t],
									   node.params["yMax"][t]])
						node_ind.append(ind)

				# Next consider all handshake boxes at time t
				for bbox in handshake_data[str(t)]:
					bb_hs = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
					idx = bbox["id"]

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
					shakes[idx][t] = iou

			unclassified = shakes.pop(-1)	# non-id shakes

			print(shakes)

			for idx in shakes:
				shake_t = shakes[idx].keys()
				shake_iou = list(shakes[idx].values())

				shakes_iou_avg = np.mean(np.array(shake_iou), axis=0).astype(float)

				# print(shakes_iou_avg)

				p1, p2 = np.argpartition(shakes_iou_avg, -2)[-2:]
				p1, p2 = int(p1), int(p2)

				for t in shake_t:
					graph.nodes[p1].params["handshake"][t] = {"person": p2, "confidence": None, "iou": shakes_iou_avg[p1]}
					graph.nodes[p2].params["handshake"][t] = {"person": p1, "confidence": None, "iou": shakes_iou_avg[p2]}

		else:
			for t in handshake_frames:
				# First take all the detected nodes at time t
				node_t = []
				node_ind = []
				for ind, node in enumerate(graph.nodes):
					if node.params["detection"][t]:
						node_t.append([node.params["xMin"][t], node.params["yMin"][t], node.params["xMax"][t],
									   node.params["yMax"][t]])
						node_ind.append(ind)

				# Next consider all handshake boxes at time t
				# nbox = handshake_data[str(t)]["No of boxes"]

				# print(t, node_t)

				for bbox in handshake_data[str(t)]["bboxes"]:
					bb_hs = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
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

					# print(t, p1, p2, iou)

					graph.nodes[p1].params["handshake"][t] = {"person": p2, "confidence": conf, "iou": iou[ind1]}
					graph.nodes[p2].params["handshake"][t] = {"person": p1, "confidence": conf, "iou": iou[ind2]}

		graph.state["handshake"] = 2
		print("Updated the graph")

	def runForBatch(self):
		self.update_handshake()


if __name__ == "__main__":

	json_loc = "./data/labels/DEEE/handshake/cctv1.json"
	img_loc = "./data/videos/DEEE/cctv1.mp4"

	parser = argparse.ArgumentParser()

	parser.add_argument("--input_file", "-i", type=str, dest="input_file", default=img_loc)
	parser.add_argument("--output_file", "-o", type=str, dest="output_file", default=json_loc)
	parser.add_argument("--overwrite", "--ow", action="store_true", dest="overwrite")
	parser.add_argument("--visualize", "--vis", action="store_true", dest="visualize")
	parser.add_argument("--verbose", "--verb", action="store_true", dest="verbose")
	parser.add_argument("--tracked", "-t", type=bool, dest="tracked", default=True)

	args = parser.parse_args()
	args.overwrite = True

	img_loc = args.input_file
	json_loc = args.output_file

	# TEST
	img_handle = NNHandler_image(format="avi", img_loc=img_loc)
	img_handle.runForBatch()

	hs_handle = NNHandler_handshake(vis=args.visualize, is_tracked=args.tracked)
	try:
		if os.path.exists(json_loc):
			if args.overwrite:
				raise Exception("Overwriting json : %s"%json_loc)

			# To load YOLO + DSORT track from json
			hs_handle.init_from_json(json_loc)

		else:
			raise Exception("Json does not exists : %s"%json_loc)
	except:
		# To create YOLO + DSORT track and save to json
		hs_handle.create_yolo(img_handle)
		hs_handle.save_json(json_loc)



	# g = Graph()
	# # graph_json = './data/vid-01-graph_handshake_track.json'
	#
	# try:
	# 	if os.path.exists(graph_json):
	#
	# 		# init graph from json
	# 		g.init_from_json(graph_json)
	# 	else:
	# 		raise Exception("Json does not exists : %s"%graph_json)
	# except:
	# 	hs_handle.connectToGraph(g)
	# 	hs_handle.runForBatch()
	#
	# 	print(g)
	# 	g.saveToFile('./data/vid-01-graph_handshake_track.json')
	#
	# g.plot()

