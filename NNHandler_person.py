import argparse
import json
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from collections import defaultdict

from NNHandler_yolo import NNHandler_yolo
from NNHandler_image import NNHandler_image, cv2

from Node_Person import Person

from suren.util import Json, eprint


class NNHandler_person(NNHandler_yolo):

	weigths_filename = NNHandler_yolo.yolo_dir + '/checkpoints/yolov4-416'

	class_names = ["person"]

	# Definition of the parameters
	max_cosine_distance = 0.4
	nn_budget = None
	nms_max_overlap = 1.0

	iou_thresh = .45
	score_thresh = .5
	input_size = 416


	def __init__(self, json_file=None, is_tracked=True, vis=False, verbose=True, debug=False):

		super().__init__(json_file=json_file, is_tracked=is_tracked, vis=vis, verbose=verbose, debug=debug)
		print("\t[*] Person detector")

	def extractValForKey(self,st,startSt,endSt):
		a=st.index(startSt)+len(startSt)
		b=st.index(endSt)
		return st[a:b].strip()

	def refinePersonTrajectory(self,p):
		# @ Is this function working?? getparam needs 2 arguments
		firstApperanceT=0
		lastAppearanceT=p.timeSeriesLength-1

		for a in range(p.timeSeriesLength):
			if p.getParam("detection")==False:
				firstApperanceT=a

		for a in range(p.timeSeriesLength-1,-1,-1):
			if p.getParam("detection")==False:
				lastAppearanceT=a			


		print("This person is visible only from {} to {} frames".format(firstApperanceT,lastAppearanceT))

	def update_graph_nodes(self, start_time=None, end_time = None):
		if start_time is None: start_time = 0
		if end_time is None: end_time = self.time_series_length

		graph = self.graph
		if graph.time_series_length is None: graph.time_series_length = end_time-start_time
		else: raise Exception("Graph is not empty")

		assert len(graph.nodes) == 0, "Graph not empty. Cannot update non-empty graph"

		person_dic = defaultdict(dict)

		for t in range(start_time, end_time):
			try:
				yolo_bbox = self.json_data[t]
			except KeyError:
				try:
					yolo_bbox = self.json_data[str(t)]		# If reading from json file
				except:
					continue								# No boxes detected


			for bbox in yolo_bbox:
				idx = bbox["id"]
				person_dic[idx][t] = bbox

		if -1 in person_dic:
			unclassified = person_dic.pop(-1)

		# print(person_dic)

		for idx in sorted(person_dic):
			# TEMP SOLUTION FOR GIHAN

			detected = {t : (True if t in person_dic[idx] else False) for t in range(start_time, end_time)}

			x_min = [person_dic[idx][t]["x1"] if detected[t] else 0 for t in range(start_time, end_time)]
			x_max = [person_dic[idx][t]["x2"] if detected[t] else 0 for t in range(start_time, end_time)]
			y_min = [person_dic[idx][t]["y1"] if detected[t] else 0 for t in range(start_time, end_time)]
			y_max = [person_dic[idx][t]["y2"] if detected[t] else 0 for t in range(start_time, end_time)]

			detected = [detected[t] for t in detected]

			p = Person(time_series_length=end_time-start_time,
					   initParams={"id":idx, "xMin":x_min, "xMax":x_max, "yMin":y_min, "yMax":y_max, "detection":detected})
			# print(idx, p.params)
			graph.add_person(p)

		graph.state["people"] = 2


	def runForBatch(self, start_time=None, end_time = None):
		self.update_graph_nodes(start_time, end_time)




if __name__=="__main__":

	json_loc = "./data/labels/DEEE/yolo/cctv3-yolo.json"
	img_loc = "./data/videos/DEEE/cctv3.mp4"

	parser = argparse.ArgumentParser()

	parser.add_argument("--input_file", "-i", type=str, dest="input", default=img_loc)
	parser.add_argument("--output_file", "-o", type=str, dest="output", default=json_loc)
	parser.add_argument("--overwrite", "--ow", action="store_true", dest="overwrite")
	parser.add_argument("--visualize", "--vis", action="store_true", dest="visualize")
	parser.add_argument("--verbose", "--verb", action="store_true", dest="verbose")
	parser.add_argument("--tracked", "-t", type=bool, dest="tracked", default=True)

	args = parser.parse_args()

	args.input = "./data/videos/TownCentreXVID.mp4"
	args.output = "./data/labels/TownCentre/person_5.json"
	args.overwrite = False
	args.verbose=True
	args.visualize=True

	img_loc = args.input
	json_loc = args.output

	# TEST
	img_handle = NNHandler_image(format="avi", img_loc=img_loc)
	img_handle.runForBatch()

	person_handler = NNHandler_person(json_file=json_loc, vis=args.visualize, is_tracked=args.tracked, verbose=args.verbose, debug=False)
	if os.path.exists(json_loc) and not args.overwrite:
		# To load YOLO + DSORT track from json
		person_handler.init_from_json()
	else:
		# To create YOLO + DSORT track and save to json
		person_handler.create_yolo(img_handle)
		person_handler.save_json()
