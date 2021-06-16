import numpy as np
import json
import os, sys
# from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse

from NNHandler import NNHandler
from NNHandler_yolo import  NNHandler_yolo
from NNHandler_image import NNHandler_image, cv2
from Graph import Graph

from suren.util import get_iou, Json, eprint

# This is only needed if running YOLO / deepsort
# Not needed if the values are loaded from file

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


class NNHandler_mask(NNHandler_yolo):
	weigths_filename = NNHandler_yolo.yolo_dir + '/checkpoints/yolov4-obj_best'

	class_names = ["Mask", "NoMask"]

	# Definition of the parameters
	max_cosine_distance = 0.4
	nn_budget = None
	nms_max_overlap = 1.0

	iou_thresh = .45
	score_thresh = .2
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

	def __init__(self, mask_file=None, is_tracked=False, vis=False, verbose=False):

		super().__init__()

		print("Creating a mask handler")
		self.json_file = mask_file
		self.is_tracked = is_tracked
		self.visualize = vis
		self.verbose = verbose

		print(self)

	def __repr__(self):
		lines = []
		if self.json_file is not None:
			lines.append("\t[*] Json location : %s" % self.json_file)
		lines.append("\t[*] Tracked : {}".format(self.is_tracked))
		return "\n".join(lines)

	def runForBatch(self):
		self.init_from_json()


if __name__ == "__main__":

	img_loc = "./suren/temp/18.avi"
	json_loc = "./data/vid-01-mask.json"

	parser = argparse.ArgumentParser()

	parser.add_argument("--input", "-i", type=str, dest="input", default=img_loc)
	parser.add_argument("--output", "-o", type=str, dest="output", default=json_loc)

	parser.add_argument("--overwrite", "--ow", action="store_true", dest="overwrite")
	parser.add_argument("--visualize", "--vis", action="store_true", dest="visualize")
	parser.add_argument("--verbose", "--verb", action="store_true", dest="verbose")
	parser.add_argument("--tracker", "-t", action="store_false", dest="tracker")

	args = parser.parse_args()

	img_loc = args.input
	json_loc = args.output

	args.visualize=False
	args.verbose=True
	args.tracker=False


	# TEST
	img_handle = NNHandler_image(format="avi", img_loc=img_loc)
	img_handle.runForBatch()

	nn_handle = NNHandler_mask(mask_file=json_loc, is_tracked=args.tracker, vis=args.visualize)

	try:
		if os.path.exists(json_loc):
			if args.overwrite:
				raise Exception("Overwriting json : %s" % json_loc)

			# To load YOLO + DSORT track from json
			nn_handle.init_from_json()

		else:
			raise Exception("Json does not exists : %s" % json_loc)
	except:
		# To create YOLO mask + DSORT track and save to json
		nn_handle.create_yolo(img_handle)
		nn_handle.save_json()


	# g = Graph()
	# g.init_from_json('./data/vid-01-graph.json')

	# # init graph from json
	# try:
	# 	g.init_from_json('./data/vid-01-graph_mask_track.json')
	# except:
	# 	nn_handle.connectToGraph(g)
	# 	nn_handle.runForBatch()
	#
	# 	print("Created graph with nodes = %d for frames = %d. Param example:" % (g.n_nodes, g.time_series_length))
	# 	for p in g.nodes[0].params:
	# 		print(p, g.nodes[0].params[p])
	#
	# 	g.saveToFile('./data/vid-01-graph_mask_track.json')

	# g.plot()

