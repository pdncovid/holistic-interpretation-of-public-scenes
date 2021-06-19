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
	sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/submodules/yolov4-deepsort")

	import tensorflow as tf
	from tensorflow.python.saved_model import tag_constants

	from deep_sort import preprocessing, nn_matching
	from deep_sort.detection import Detection
	from deep_sort.tracker import Tracker
	from tools import generate_detections as gdet
	import core.utils as utils

	# NOT NEEDED in this code
	# from core.yolov4 import filter_boxes
	# from core.config import cfg

except Exception as e:
	eprint("Cannot run YOLO:", e)

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


class NNHandler_yolo(NNHandler):
	yolo_dir = os.path.dirname(os.path.realpath(__file__)) + "/model/yolov4-deepsort"

	model_filename = yolo_dir + '/model_data/mars-small128.pb'
	weigths_filename = yolo_dir + '/checkpoints/yolov4-416'

	class_names = None

	# Definition of the parameters
	max_cosine_distance = 0.4
	nn_budget = None
	nms_max_overlap = 1.0

	iou_thresh = .45
	score_thresh = .5
	input_size = 416

	@staticmethod
	def YOLO_import():
		raise NotImplementedError

	@staticmethod
	def get_parse():
		parser = argparse.ArgumentParser()

		parser.add_argument("--input_file", "-i", type=str, dest="input_file", default=None)
		parser.add_argument("--output_file", "-o", type=str, dest="output_file", default=None)

		parser.add_argument("--overwrite", "--ow", action="store_true", dest="overwrite")
		parser.add_argument("--visualize", "--vis", action="store_true", dest="visualize")
		parser.add_argument("--verbose", "--verb", action="store_true", dest="verbose")
		parser.add_argument("--tracked", "-t", type=bool, dest="tracked", default=True)

		args = parser.parse_args()
		return args

	@staticmethod
	def plot(img, bbox, col):
		cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), tuple(col), 2)

		if len(bbox) > 4:
			cv2.putText(img, str(bbox[4]), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75, (255, 255, 255), 2)

	# @staticmethod
	# def plot(img, points, col):
	# 	x_min, y_min, x_max, y_max = points
	# 	cv2.rectangle(img, (x_min, y_min), (x_max, y_max), col, 2)

	def __init__(self, json_file=None, is_tracked=True, vis=True, verbose=True, debug=False):

		super().__init__()

		print("Creating a YOLO handler")

		self.json_file = json_file
		self.is_tracked = is_tracked
		self.visualize = vis
		self.verbose = verbose
		self.debug = debug

	def create_yolo(self, img_handle):
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
		nms_max_overlap = self.nms_max_overlap
		iou_thresh = self.iou_thresh
		score_thresh = self.score_thresh
		input_size = self.input_size

		# YOLO encoder
		encoder = gdet.create_box_encoder(self.model_filename, batch_size=1)

		# initialize deep sort
		if self.is_tracked:
			max_cosine_distance = self.max_cosine_distance
			nn_budget = self.nn_budget

			# calculate cosine distance metric
			metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
			# initialize tracker
			tracker = Tracker(metric, n_init=3)
		else:
			tracker = None

		saved_model_loaded = tf.saved_model.load(self.weigths_filename, tags=[tag_constants.SERVING])
		infer = saved_model_loaded.signatures['serving_default']

		frame_num = 0
		img_handle.open()
		for t in range(img_handle.time_series_length):

			frame = img_handle.read_frame(t)
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			frame_num += 1

			if self.verbose: print('Frame #: ', frame_num)
			# if t < 1000: continue

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

			# WTF : why a loop above???
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

			if self.debug:
				print("[xx]", pred_bbox)

			# Give class names
			if self.class_names is None: self.class_names = []
			# names = [self.class_names[int(i)] if int(i) < len(self.class_names) else str(i) for i in classes]
			try:
				names = [self.class_names[int(i)] for i in classes]
			except:
				names = [self.class_names[int(i)] if int(i) < len(self.class_names) else "class_%d"%i for i in classes]
				eprint("[xx]", classes)

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
			if self.is_tracked:
				tracker.predict()
				tracker.update(detections)

				detections = tracker.tracks

			person_t = []

			# update tracks
			for track in detections:

				# Get confidence (@suren : Don't think this is needed. Just uncomment/delet)

				if self.is_tracked and (not track.is_confirmed() or track.time_since_update > 1):
					if not track.is_confirmed():
						bbox = track.to_tlbr()
						person_t.append({
							"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3], "id": -1
						})

					continue

				bbox = track.to_tlbr()
				class_name = track.get_class()

				# draw bbox on screen
				if self.is_tracked:
					id = track.track_id
					txt = class_name + "-" + str(track.track_id)
				else:
					id = -1
					txt = class_name

				color = colors[int(id) % len(colors)]
				color = [i * 255 for i in color]

				cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
				cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)),
							  (int(bbox[0]) + len(txt) * 17, int(bbox[1])), color, -1)
				cv2.putText(frame, txt, (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75, (255, 255, 255), 2)

				# if enable info flag then print details about each track
				if self.verbose:
					print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(
						str(id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

				person_t.append({
					"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3], "id": id
				})

			# result = np.asarray(frame)
			result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

			if self.visualize:
				cv2.imshow("Output Video", result)
				if cv2.waitKey(20) & 0xFF == ord('q'): break

			if len(person_t) > 0: tracked_person[t] = person_t

		if self.visualize:
			cv2.destroyAllWindows()

		self.time_series_length = frame_num
		self.json_data = tracked_person

	def init_from_json(self, file_name=None):
		if file_name is None: file_name = self.json_file
		if file_name is None or not os.path.exists(file_name): raise ValueError("Json File does not exists : %s"%file_name)

		if self.verbose:
			print("\t[*] Init from file : %s"%file_name)

		with open(file_name, 'r') as json_file:
			data = json.load(json_file)

		self.time_series_length = data.pop("frames")
		self.json_data = data

		self.ftype = "json"

		return self.json_data

	def save_json(self, file_name=None):
		if file_name is None: file_name = self.json_file
		if not os.path.exists(os.path.dirname(file_name)) : os.makedirs(os.path.dirname(file_name))

		js = Json(file_name)
		dic = {"frames": self.time_series_length}
		for i in self.json_data:
			dic[i] = self.json_data[i]

		self.ftype = "json"

		js.write(dic)



'''
if __name__=="__main__":

	img_loc = "./suren/temp/seq18.avi"
	json_loc = "./data/vid-01-yolo.json"

	parser = argparse.ArgumentParser()

	parser.add_argument("--nnout_yolo", "-y", type=str, dest="nnout_yolo", default=json_loc)
	parser.add_argument("--video_file", "-v", type=str, dest="video_file", default=img_loc)
	parser.add_argument("--overwrite", "-ow", action="store_true", dest="overwrite")
	parser.add_argument("--visualize", "--vis", action="store_true", dest="visualize")
	parser.add_argument("--verbose", "--verb", action="store_true", dest="verbose")
	parser.add_argument("--tracked", "-t", type=bool, dest="tracked", default=True)

	args = parser.parse_args()

	img_loc = args.video_file
	json_loc = args.nnout_yolo

	# TEST
	img_handle = NNHandler_image(format="avi", img_loc=img_loc)
	img_handle.runForBatch()

	nn_yolo = NNHandler_yolo(vis=args.visualize, is_tracked=args.tracked)
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
		nn_yolo.create_yolo(img_handle)
		nn_yolo.save_json(json_loc)

'''
