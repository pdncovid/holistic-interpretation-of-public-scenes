import argparse
import json
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from collections import defaultdict

from NNHandler import NNHandler
from NNHandler_image import NNHandler_image, cv2

from Node_Person import Person

from suren.util import Json, eprint

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
	eprint("Cannot run YOLO:", e)


class NNHandler_yolo(NNHandler):
	yolo_dir = "./suren/temp/yolov4-deepsort-master/"
	model_filename = yolo_dir + 'model_data/mars-small128.pb'
	weigths_filename = yolo_dir + '/checkpoints/yolov4-416'

	# Definition of the parameters
	max_cosine_distance = 0.4
	nn_budget = None
	nms_max_overlap = 1.0

	iou_thresh = .45
	score_thresh = .5
	input_size = 416

	@staticmethod
	def plot(img, points, col):
		x_min, y_min, x_max, y_max = points
		cv2.rectangle(img, (x_min, y_min), (x_max, y_max), col, 2)

	def __init__(self, json_file=None, is_tracked=True, vis=True, verbose=True):
		#TODO : @gihan - remove textFileName and everything related to it :)

		super().__init__()

		print("Creating a YOLO handler")

		# self.fileName=textFileName
		# self.inputBlockSize=N

		self.json_file = json_file
		self.is_tracked = is_tracked
		self.visualize = vis
		self.verbose = verbose

	def create_yolo(self, img_handle):
		"""
		:param img_handle: NNHandler_image
		:return:
		"""

		# if not import_tracker(): raise Exception("Couldn't create tracker")
		if not os.path.exists(self.yolo_dir): raise Exception("Couldn't find yolo_directory : %s" % (self.yolo_dir))
		if not os.path.exists(self.weigths_filename): raise Exception("Couldn't find weights : %s" % (self.weigths_filename))
		if not os.path.exists(self.model_filename): raise Exception("Couldn't find model : %s" % (self.model_filename))

		tracked_person = {}

		# Definition of the parameters
		nms_max_overlap = self.nms_max_overlap

		iou_thresh = self.iou_thresh
		score_thresh = self.score_thresh
		input_size = self.input_size

		# initialize deep sort
		encoder = gdet.create_box_encoder(self.model_filename, batch_size=1)

		saved_model_loaded = tf.saved_model.load(self.weigths_filename, tags=[tag_constants.SERVING])
		infer = saved_model_loaded.signatures['serving_default']

		frame_num = 0
		img_handle.open()
		for t in range(img_handle.time_series_length):
			# if t == 1000: break

			frame = img_handle.read_frame(t)
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			# image = Image.fromarray(frame)
			frame_num += 1
			print('Frame #: ', frame_num)
			frame_size = frame.shape[:2]
			image_data = cv2.resize(frame, (input_size, input_size))
			image_data = image_data / 255.
			image_data = image_data[np.newaxis, ...].astype(np.float32)

			# run detections on tflite if flag is set

			batch_data = tf.constant(image_data)
			pred_bbox = infer(batch_data)
			for key, value in pred_bbox.items():
				boxes = value[:, :, 0:4]
				pred_conf = value[:, :, 4:]

				print(boxes, pred_conf)

			boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
				boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
				scores=tf.reshape(
					pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
				max_output_size_per_class=50,
				max_total_size=50,
				iou_threshold=iou_thresh,
				score_threshold=score_thresh
			)

			# convert data to numpy arrays and slice out unused elements
			num_objects = valid_detections.numpy()[0]
			bboxes = boxes.numpy()[0]
			bboxes = bboxes[0:int(num_objects)]
			scores = scores.numpy()[0]
			scores = scores[0:int(num_objects)]
			classes = classes.numpy()[0]
			classes = classes[0:int(num_objects)]

			print(num_objects, bboxes, scores, classes)

			# format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
			original_h, original_w, _ = frame.shape
			bboxes = utils.format_boxes(bboxes, original_h, original_w)

			# store all predictions in one parameter for simplicity when calling functions
			pred_bbox = [bboxes, scores, classes, num_objects]

			print(pred_bbox)

			class_names = ["Mask", "No_Mask"]
			names = [class_names[int(i)] for i in classes]

			# encode yolo detections and feed to tracker
			features = encoder(frame, bboxes)
			detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
						  zip(bboxes, scores, names, features)]

			# initialize color map
			cmap = plt.get_cmap('tab20b')
			colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

			# run non-maxima supression
			boxs = np.array([d.tlwh for d in detections])
			scores = np.array([d.confidence for d in detections])
			classes = np.array([d.class_name for d in detections])

			indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
			detections = [detections[i] for i in indices]

			# print(detections)

			person_t = []

			# update tracks
			for id, track in enumerate(detections):

				bbox = track.to_tlbr()
				class_name = track.get_class()

				# draw bbox on screen
				color = colors[id % len(colors)]
				color = [i * 255 for i in color]
				cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
				cv2.putText(frame, class_name, (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75, color, 2)

				# if enable info flag then print details about each track
				if self.verbose:
					print("Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(
						class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

				person_t.append({
					"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3], "id": -1
				})

			result = np.asarray(frame)
			result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

			if self.visualize:
				cv2.imshow("Output Video", result)
				if cv2.waitKey(20) & 0xFF == ord('q'): break

			if len(person_t) > 0: tracked_person[t] = person_t

		if self.visualize:
			cv2.destroyAllWindows()

		self.time_series_length = frame_num
		self.json_data = tracked_person


	def create_tracker(self, img_handle):
		"""
		:param img_handle: NNHandler_image
		:return:
		"""

		# if not import_tracker(): raise Exception("Couldn't create tracker")
		if not os.path.exists(self.yolo_dir): raise Exception("Couldn't find yolo_directory : %s" % (self.yolo_dir))
		if not os.path.exists(self.model_filename): raise Exception("Couldn't find model : %s" % (self.model_filename))
		if not os.path.exists(self.weigths_filename): raise Exception("Couldn't find weights : %s" % (self.weigths_filename))

		tracked_person = {}

		# Definition of the parameters
		max_cosine_distance = self.max_cosine_distance
		nn_budget = self.nn_budget
		nms_max_overlap = self.nms_max_overlap

		iou_thresh = self.iou_thresh
		score_thresh = self.score_thresh
		input_size = self.input_size

		# initialize deep sort
		encoder = gdet.create_box_encoder(self.model_filename, batch_size=1)
		# calculate cosine distance metric
		metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
		# initialize tracker
		tracker = Tracker(metric, n_init=3)

		saved_model_loaded = tf.saved_model.load(self.weigths_filename, tags=[tag_constants.SERVING])
		infer = saved_model_loaded.signatures['serving_default']

		frame_num = 0
		img_handle.open()
		for t in range(img_handle.time_series_length):
			# if t == 1000: break

			frame = img_handle.read_frame(t)
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			# image = Image.fromarray(frame)
			frame_num += 1
			print('Frame #: ', frame_num)
			frame_size = frame.shape[:2]
			image_data = cv2.resize(frame, (input_size, input_size))
			image_data = image_data / 255.
			image_data = image_data[np.newaxis, ...].astype(np.float32)

			# run detections on tflite if flag is set

			batch_data = tf.constant(image_data)
			pred_bbox = infer(batch_data)
			for key, value in pred_bbox.items():
				boxes = value[:, :, 0:4]
				pred_conf = value[:, :, 4:]

			# print(boxes, pred_conf)

			boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
				boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
				scores=tf.reshape(
					pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
				max_output_size_per_class=50,
				max_total_size=50,
				iou_threshold=iou_thresh,
				score_threshold=score_thresh
			)

			# convert data to numpy arrays and slice out unused elements
			num_objects = valid_detections.numpy()[0]
			bboxes = boxes.numpy()[0]
			bboxes = bboxes[0:int(num_objects)]
			scores = scores.numpy()[0]
			scores = scores[0:int(num_objects)]
			classes = classes.numpy()[0]
			classes = classes[0:int(num_objects)]

			# print(num_objects, bboxes, scores, classes)

			# format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
			original_h, original_w, _ = frame.shape
			bboxes = utils.format_boxes(bboxes, original_h, original_w)

			# store all predictions in one parameter for simplicity when calling functions
			pred_bbox = [bboxes, scores, classes, num_objects]

			# print(pred_bbox)

			names = ['person'] * num_objects

			# encode yolo detections and feed to tracker
			features = encoder(frame, bboxes)
			detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
						  zip(bboxes, scores, names, features)]

			# initialize color map
			cmap = plt.get_cmap('tab20b')
			colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

			# run non-maxima supression
			boxs = np.array([d.tlwh for d in detections])
			scores = np.array([d.confidence for d in detections])
			classes = np.array([d.class_name for d in detections])

			indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
			detections = [detections[i] for i in indices]

			# print(detections)

			# Call the tracker
			tracker.predict()
			tracker.update(detections)

			person_t = []

			# update tracks
			for track in tracker.tracks:

				# Get confidence (@suren : Don't think this is needed. Just uncomment/delet)

				if not track.is_confirmed() or track.time_since_update > 1:
					if not track.is_confirmed():
						bbox = track.to_tlbr()
						person_t.append({
							"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3], "id": -1
						})

					continue

				bbox = track.to_tlbr()
				class_name = track.get_class()

				# draw bbox on screen
				color = colors[int(track.track_id) % len(colors)]
				color = [i * 255 for i in color]
				cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
				cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)),
							  (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])), color,
							  -1)
				cv2.putText(frame, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
							(255, 255, 255), 2)

				# if enable info flag then print details about each track
				if self.verbose:
					print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(
						str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

				person_t.append({
					"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3], "id": track.track_id
				})

			result = np.asarray(frame)
			result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

			if self.visualize:
				cv2.imshow("Output Video", result)
				if cv2.waitKey(20) & 0xFF == ord('q'): break

			if len(person_t) > 0: tracked_person[t] = person_t

		if self.visualize:
			cv2.destroyAllWindows()

		self.time_series_length = frame_num
		self.json_data = tracked_person

	
	def extractValForKey(self,st,startSt,endSt):
		a=st.index(startSt)+len(startSt)
		b=st.index(endSt)
		return st[a:b].strip()

	def init_from_json(self, file_name=None):
		if file_name is None: file_name = self.json_file

		with open(file_name) as json_file:
			data = json.load(json_file)

		self.time_series_length = data.pop("frames")
		self.json_data = data

		self.ftype = "json"

		return self.json_data

	def save_json(self, json_file=None):
		if json_file is None: json_file = self.json_file

		js = Json(json_file)
		dic = {"frames": self.time_series_length}
		for i in self.json_data:
			dic[i] = self.json_data[i]

		self.ftype = "json"

		js.write(dic)

	def refinePersonTrajectory(self,p):
		firstApperanceT=0
		lastAppearanceT=p.timeSeriesLength-1

		for a in range(p.timeSeriesLength):
			if p.getParam("detection")==False:
				firstApperanceT=a

		for a in range(p.timeSeriesLength-1,-1,-1):
			if p.getParam("detection")==False:
				lastAppearanceT=a			


		print("This person is visible only from {} to {} frames".format(firstApperanceT,lastAppearanceT))

	def update_graph_nodes(self):
		if self.graph.time_series_length is None: self.graph.time_series_length = self.time_series_length

		person_dic = defaultdict(dict)

		for t in range(self.time_series_length):
			try: yolo_bbox = self.json_data[t]
			except: yolo_bbox = self.json_data[str(t)]		# If reading from json file

			for bbox in yolo_bbox:
				idx = bbox.pop("id")
				person_dic[idx][t] = bbox

		unclassified = person_dic.pop(-1)

		# print(person_dic)

		for idx in person_dic:
			detected = [True if t in person_dic[idx] else False for t in range(self.time_series_length)]

			x_min = [person_dic[idx][t]["x1"] if detected[t] else 0 for t in range(self.time_series_length)]
			x_max = [person_dic[idx][t]["x2"] if detected[t] else 0 for t in range(self.time_series_length)]
			y_min = [person_dic[idx][t]["y1"] if detected[t] else 0 for t in range(self.time_series_length)]
			y_max = [person_dic[idx][t]["y2"] if detected[t] else 0 for t in range(self.time_series_length)]

			p = Person(time_series_length=self.time_series_length,
					   initParams={"xMin":x_min, "xMax":x_max, "yMin":y_min, "yMax":y_max, "detection":detected})
			# print(idx, p.params)
			self.graph.add_person(p)


	def runForBatch(self):
		print("Running Yolo handler for batch....")
		# if self.ftype == "txt":
		# 	print("Running Yolo handler for batch....")
		# 	frames=[]
		# 	for l in self.allLines[1:]:
		# 		if l.split(" ")[0]=="Frame":
		# 			frames.append([])
		# 		elif l.split(" ")[0]=="FPS:":
		# 			pass
		# 		elif l.split(" ")[0]=="Tracker":
		# 			# print()
		# 			try:
		# 				a="Tracker ID:"
		# 				b="Class:"
		# 				c="BBox Coords (xmin, ymin, xmax, ymax):"
		# 				d="\n"
		# 				frames[-1].append(dict())
		# 				# print(frames[-1])
		# 				frames[-1][-1]["id"]=int(self.extractValForKey(l,a,b)[:-1])
		# 				frames[-1][-1]["class"]=self.extractValForKey(l,b,c)
		# 				frames[-1][-1]["bbox"]=list(map(int,self.extractValForKey(l,c,d)[1:-1].split(",")))
		# 			except:
		# 				break
		# 		else:
		# 			print("Unidentified line: ",l)
		#
		# 	print(len(frames))
		# 	ids=[]
		# 	for f in frames:
		# 		for o in f:
		# 			ids.append(o["id"])
		# 	ids=sorted(set(ids))
		#
		# 	print("UniqueIDs ",ids)
		#
		#
		# 	for i in range(len(ids)):
		# 		self.graph.add_person()
		# 		node=self.graph.getNode(i)
		# 		# node.addParam("detection")
		# 		for t in range(len(frames)):
		# 			node.setParam("xMin",t,0)
		# 			node.setParam("yMin",t,0)
		# 			node.setParam("xMax",t,0)
		# 			node.setParam("yMax",t,0)
		# 			node.setParam("detection",t,False)
		# 			for pt in range(len(frames[t])):
		# 				if frames[t][pt]["id"]==ids[i]:
		#
		#
		# 					node.setParam("xMin",t,frames[t][pt]["bbox"][0])
		# 					node.setParam("xMax",t,frames[t][pt]["bbox"][2])
		# 					node.setParam("yMin",t,frames[t][pt]["bbox"][1])
		# 					node.setParam("yMax",t,frames[t][pt]["bbox"][3])
		# 					node.setParam("detection",t,True)
		#
		#
		# 	# self.graph.saveToFile(fileName="yoloExp.txt")
		# 	# self.myInput()
		#
		#
		# 	# print(self.allLines)
		#
		# 	print("Updated the graph")

		self.init_from_json()
		self.update_graph_nodes()




if __name__=="__main__":

	img_loc = "./suren/temp/seq18.avi"
	json_loc = "./data/vid-01-yolo.json"

	parser = argparse.ArgumentParser()

	parser.add_argument("--nnout_yolo", "-y", type=str, dest="nnout_yolo", default=json_loc)
	parser.add_argument("--video_file", "-v", type=str, dest="video_file", default=img_loc)
	parser.add_argument("--overwrite", "-ow", action="store_true", dest="overwrite")
	parser.add_argument("--visualize", "--vis", action="store_true", dest="visualize")
	parser.add_argument("--verbose", "--verb", action="store_true", dest="verbose")

	args = parser.parse_args()

	img_loc = args.video_file
	json_loc = args.nnout_yolo


	# TEST
	img_handle = NNHandler_image(format="avi", img_loc=img_loc)
	img_handle.runForBatch()

	nn_yolo = NNHandler_yolo(vis=args.visualize)
	try:
		if os.path.exists(json_loc):
			if args.overwrite:
				raise Exception("Overwriting json : %s"%json_loc)

			# To load YOLO + DSORT track from json
			nn_yolo.init_from_json(json_loc)

		else:
			raise Exception("Json does not exists : %s"%json_loc)
	except:
		# To create YOLO + DSORT track and save to json
		nn_yolo.create_tracker(img_handle)
		nn_yolo.save_json(json_loc)


