import numpy as np
import json
import os, sys
# from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
from glob import glob

from NNHandler import NNHandler
from NNHandler_image import NNHandler_image, cv2
from Graph import Graph

from suren.util import get_iou, Json, eprint

# This is only needed if running YOLO / deepsort
# Not needed if the values are loaded from file
# try:
# 	import tensorflow as tf
# 	from tensorflow.python.saved_model import tag_constants
#
# 	sys.path.append(os.path.relpath('./suren/temp/yolov4-deepsort-master'))
#
# 	from deep_sort import preprocessing, nn_matching
# 	from deep_sort.detection import Detection
# 	from deep_sort.tracker import Tracker
# 	from tools import generate_detections as gdet
# 	import core.utils as utils
# 	# from core.yolov4 import filter_boxes
# 	from tensorflow.python.saved_model import tag_constants
#
# 	from core.config import cfg
# except Exception as e:
# 	print(e)
# 	print("If YOLO isn't required, ignore this")

def import_tracker(name="sort"):
	if name == "deepsort":
		try:

			from deep_sort.tracker import Tracker, nn_matching
			from deep_sort.detection import Detection
			from deep_sort.tracker import Tracker
			from tools import generate_detections as gdet
			return True
		except:
			eprint("Deepsort not installed.")
			return False

	elif name == "sort":

		sys.path.append(os.path.relpath('./suren/temp/sort_master'))
		import sort, iou_batch		# TODO : Make this path independent

	else:
		raise NotImplementedError


class NNHandler_openpose(NNHandler):
	VERSION = "1.3"
	TRACKER = "SORT"
	MODEL = "BODY25"
	BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
				  "LShoulder": 5, "LElbow": 6, "LWrist": 7, "Hip": 8, "RHip": 9, "RKnee": 10,
				  "RAnkle": 11, "LHip": 12, "LKnee": 13, "LAnkle": 14, "REye": 15,
				  "LEye": 16, "REar": 17, "LEar": 18, "LSole": 19, "LToe": 20,
				  "LHeel": 21, "RSole": 22, "RToe": 23, "RHeel": 24, "Background": 25}

	POSE_PAIRS = [["Neck", "RShoulder"], ["RShoulder", "RElbow"], ["RElbow", "RWrist"],
				  ["Neck", "LShoulder"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
				  ["Neck", "Hip"], ["Neck", "Nose"],
				  ["Nose", "REye"], ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"],
				  ["Hip", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"],
				  ["RAnkle", "RHeel"], ["RAnkle", "RSole"], ["RSole", "RToe"],
				  ["Hip", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"],
				  ["LAnkle", "LHeel"], ["LAnkle", "LSole"], ["LSole", "LToe"],
				  ]
	THRESH = 1e-2

	@staticmethod
	def list2points(points):
		return [(points[3 * i], points[3 * i + 1], points[3 * i + 2]) for i in range(len(points) // 3)]


	@staticmethod
	def plot(img, data, is_tracked):

		bbox = []

		for p in data:
			points = NNHandler_openpose.list2points(p['pose_keypoints_2d'])

			NNHandler_openpose.plot_points(img, points)
			if is_tracked:
				points = np.array([[x, y] for (x, y, c) in points if c > NNHandler_openpose.THRESH]).reshape(-1, 2)
				x_min, y_min = np.min(points, axis=0)
				x_max, y_max = np.max(points, axis=0)
				bbox.append([x_min, y_min, x_max, y_max, p["id"]])

		if is_tracked:
			NNHandler_openpose.draw_bbox(img, bbox, col=np.zeros((25, 3)))


	@staticmethod
	def plot_points(img, points):
		if not isinstance(points, list):
			raise NotImplementedError("Handle not list types : %s"%type(points))
		elif len(points) != 25:
			points = NNHandler_openpose.list2points(points)

		for (x, y, c) in points:
			if c > NNHandler_openpose.THRESH:
				cv2.circle(img, (int(x), int(y)), 2, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
			# if i%5== 0:
			# 	cv2.putText(img, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, lineType=cv2.LINE_AA)

		for [p1, p2] in NNHandler_openpose.POSE_PAIRS:
			x1, y1, c1 = points[NNHandler_openpose.BODY_PARTS[p1]]
			x2, y2, c2 = points[NNHandler_openpose.BODY_PARTS[p2]]
			if c1 > NNHandler_openpose.THRESH and c2 > NNHandler_openpose.THRESH:
				cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

	@staticmethod
	def draw_bbox(img, bboxes, col):
		for idx, bbox in enumerate(bboxes):
			cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), tuple(col[idx]), 2)
			# cv2.rectangle(img, (int(bbox[0]), int(bbox[1] - 30)),
			# 			  (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])), color,
			# 			  -1)

			if len(bbox) > 4:
				track_id = bbox[4]
				cv2.putText(img, str(track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75, (255, 255, 255), 2)

	def __init__(self, openpose_dir=None, openpose_file=None, is_tracked=False):

		super().__init__()

		# TODO : @Suern, insert tracking data (STAF)
		# assert is_tracked == False

		print("Creating an Openpose handler")
		self.openpose_dir = openpose_dir
		self.openpose_file = openpose_file
		self.is_tracked = is_tracked
		self.json_data = None

	def init_from_json(self, openpose_file=None):
		openpose_file = self.openpose_file if openpose_file is None else openpose_file

		if openpose_file is not None:
			with open(openpose_file) as json_file:
				data = json.load(json_file)

			self.time_series_length = data.pop("frames")
			self.json_data = data

		elif self.openpose_dir is not None:
			openpose_files = list(map(lambda x: x.replace('\\', '/'), glob(self.openpose_dir + "/*.json")))

			json_data = {}

			for t, file in enumerate(openpose_files):
				with open(file) as json_file:
					data = json.load(json_file)

				ver = data["version"]
				data = data['people']

				# for ind, person in enumerate(data):
				# 	print(ind, person)
				# 	for key in person:
				# 		print(key, person[key])
				# input("press to continue")

				lis = [{'pose_keypoints_2d': person['pose_keypoints_2d'], 'id': person['person_id']} for person in data]

				json_data[t] = lis

			self.time_series_length = len(openpose_files)
			self.json_data = json_data

		else:
			raise ValueError("Specify openpose_file or openpose_dir")

		return self.json_data

	def save_json(self, openpose_file=None):
		if openpose_file is None: openpose_file = self.openpose_file

		js = Json(openpose_file)

		data = self.json_data
		data["frames"] = self.time_series_length

		js.write(data)

	def create_tracker(self, img_handle=None):

		print("Creating an Openpose tracker")

		assert self.json_data is not None, "json_data cannot be None. Call init_from_json()"

		self.is_tracked = True

		mot_tracker = Sort(max_age=1, min_hits=3, iou_threshold=.3)  # create instance of the SORT tracker
		colours = np.random.rand(32, 3) *255  # used only for display

		if img_handle is not None: img_handle.open()

		for t in range(self.time_series_length):
			# print(t)
			if img_handle is not None: rgb = img_handle.read_frame(t)

			dets = []

			for idx, p in enumerate(self.json_data[str(t)]):
				# print(p['pose_keypoints_2d'])
				points = NNHandler_openpose.list2points(p['pose_keypoints_2d'])
				if img_handle is not None:
					NNHandler_openpose.plot_points(rgb, points)

				# print(points)
				points = np.array([[x, y] for (x, y, c) in points if c > NNHandler_openpose.THRESH]).reshape(-1, 2)
				# print(points)
				x_min,  y_min = np.min(points, axis=0)
				x_max, y_max = np.max(points, axis=0)
				dets.append([x_min, y_min, x_max, y_max, 1])		# TODO : 5th element must be score (not 1)

			if len(dets) == 0:
				trackers = mot_tracker.update()
			else:

				# print(dets)
				trackers = mot_tracker.update(np.array(dets))
				# print(trackers)

				iou_det_track = iou_batch(trackers, dets)
				# print(iou_det_track)

				iou_id = np.argmax(iou_det_track, axis=0)		# Match tracker with pose_points
				for i in range(len(iou_id)):
					self.json_data[str(t)][i]['id'] = int(trackers[iou_id[i], 4])


			if img_handle is not None:
				NNHandler_openpose.draw_bbox(rgb, trackers,  colours)

			# DRAW
			print(t, self.json_data[str(t)])


			if img_handle is not None:
				cv2.imshow("plot", rgb)
				k = cv2.waitKey(0)
				if k & 0xff == ord('q'): break


		if img_handle is not None: img_handle.close()


#
	# # if not import_tracker(): raise Exception("Couldn't create tracker")
	# yolo_dir = "./suren/temp/yolov4-deepsort-master/"
	# if not os.path.exists(yolo_dir): raise Exception("Couldn't find yolo_directory : %s" % (yolo_dir))
	#
	# model_filename = yolo_dir + 'model_data/mars-small128.pb'
	# if not os.path.exists(model_filename): raise Exception("Couldn't find model : %s" % (model_filename))
	# weigths_filename = yolo_dir + '/checkpoints/yolov4-fullshake_best'
	# if not os.path.exists(weigths_filename): raise Exception("Couldn't find weights : %s" % (weigths_filename))
	#
	# tracked_handshake = {}
	#
	# # Definition of the parameters
	# max_cosine_distance = 0.4
	# nn_budget = None
	# nms_max_overlap = 1.0
	#
	# iou_thresh = .45
	# score_thresh = .5
	# input_size = 416
	#
	# # initialize deep sort
	# encoder = gdet.create_box_encoder(model_filename, batch_size=1)
	# # calculate cosine distance metric
	# metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
	# # initialize tracker
	# tracker = Tracker(metric, n_init=2)
	#
	# saved_model_loaded = tf.saved_model.load(weigths_filename, tags=[tag_constants.SERVING])
	# infer = saved_model_loaded.signatures['serving_default']
	#
	# frame_num = 0
	# img_handle.open()
	# for t in range(img_handle.time_series_length):
	# 	if t == 1000:
	# 		break
	#
	# 	frame = img_handle.read_frame(t)
	# 	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	# 	# image = Image.fromarray(frame)
	# 	frame_num += 1
	# 	print('Frame #: ', frame_num)
	# 	frame_size = frame.shape[:2]
	# 	image_data = cv2.resize(frame, (input_size, input_size))
	# 	image_data = image_data / 255.
	# 	image_data = image_data[np.newaxis, ...].astype(np.float32)
	#
	# 	# run detections on tflite if flag is set
	#
	# 	batch_data = tf.constant(image_data)
	# 	pred_bbox = infer(batch_data)
	# 	for key, value in pred_bbox.items():
	# 		boxes = value[:, :, 0:4]
	# 		pred_conf = value[:, :, 4:]
	#
	# 	boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
	# 		boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
	# 		scores=tf.reshape(
	# 			pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
	# 		max_output_size_per_class=50,
	# 		max_total_size=50,
	# 		iou_threshold=iou_thresh,
	# 		score_threshold=score_thresh
	# 	)
	#
	# 	# convert data to numpy arrays and slice out unused elements
	# 	num_objects = valid_detections.numpy()[0]
	# 	bboxes = boxes.numpy()[0]
	# 	bboxes = bboxes[0:int(num_objects)]
	# 	scores = scores.numpy()[0]
	# 	scores = scores[0:int(num_objects)]
	# 	classes = classes.numpy()[0]
	# 	classes = classes[0:int(num_objects)]
	#
	# 	# print(num_objects, bboxes, scores, classes)
	#
	# 	# format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
	# 	original_h, original_w, _ = frame.shape
	# 	bboxes = utils.format_boxes(bboxes, original_h, original_w)
	#
	# 	# store all predictions in one parameter for simplicity when calling functions
	# 	pred_bbox = [bboxes, scores, classes, num_objects]
	#
	# 	print(pred_bbox)
	#
	# 	names = ['Handshake'] * num_objects
	#
	# 	# encode yolo detections and feed to tracker
	# 	features = encoder(frame, bboxes)
	# 	detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
	# 				  zip(bboxes, scores, names, features)]
	#
	# 	# initialize color map
	# 	cmap = plt.get_cmap('tab20b')
	# 	colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
	#
	# 	# run non-maxima supression
	# 	boxs = np.array([d.tlwh for d in detections])
	# 	scores = np.array([d.confidence for d in detections])
	# 	classes = np.array([d.class_name for d in detections])
	#
	# 	indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
	# 	detections = [detections[i] for i in indices]
	#
	# 	print(detections)
	#
	# 	# Call the tracker
	# 	tracker.predict()
	# 	tracker.update(detections)
	#
	# 	handshake_t = []
	#
	# 	# update tracks
	# 	for track in tracker.tracks:
	#
	# 		# Get confidence (@suren : Don't think this is needed. Just uncomment/delet)
	# 		# ind = [det.to_tlbr() for det in detections]
	# 		# conf =
	#
	# 		if not track.is_confirmed() or track.time_since_update > 1:
	# 			if not track.is_confirmed():
	# 				bbox = track.to_tlbr()
	# 				handshake_t.append({
	# 					"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3], "id": -1
	# 				})
	#
	# 			continue
	#
	# 		bbox = track.to_tlbr()
	# 		class_name = track.get_class()
	#
	# 		# draw bbox on screen
	# 		color = colors[int(track.track_id) % len(colors)]
	# 		color = [i * 255 for i in color]
	# 		cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
	# 		cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)),
	# 					  (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])), color,
	# 					  -1)
	# 		cv2.putText(frame, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
	# 					(255, 255, 255), 2)
	#
	# 		# if enable info flag then print details about each track
	# 		print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id),
	# 																							class_name, (
	# 																								int(bbox[0]),
	# 																								int(bbox[1]),
	# 																								int(bbox[2]),
	# 																								int(bbox[3]))))
	#
	# 		handshake_t.append({
	# 			"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3], "id": track.track_id
	# 		})
	#
	# 	# calculate frames per second of running detections
	# 	result = np.asarray(frame)
	# 	result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
	#
	# 	cv2.imshow("Output Video", result)
	#
	# 	if cv2.waitKey(20) & 0xFF == ord('q'): break
	#
	# 	if len(handshake_t) > 0: tracked_handshake[t] = handshake_t
	# 	max_t = t
	#
	# cv2.destroyAllWindows()
	#
	# self.time_series_length = max_t
	#
	# self.json_data = tracked_handshake

	def update_handshake(self, handshake_file=None):
		pass

	# # Use self.graph and find the two people using maximum intersection area
	#
	# handshake_data = self.json_data
	#
	# # assert self.time_series_length == self.graph.time_series_length, \
	# # 	"Both files (yolo and graph) must be of same length :/ (%d, %d)" % (
	# # 	self.time_series_length, self.graph.time_series_length)
	#
	# # This is going to be inefficient:
	# # Graph contains nodes which have time series info for separate nodes
	# # YOLO output has timeseries info first and then info of each node for that time series
	#
	# handshake_frames = list(map(int, list(handshake_data.keys())))  # write in a better way
	# # print(handshake_frames)
	#
	# if self.is_tracked:
	# 	shakes = defaultdict(dict)
	#
	# 	for t in handshake_frames:
	#
	# 		# First take all the detected nodes at time t
	# 		node_t = []
	# 		node_ind = []
	# 		for ind, node in enumerate(self.graph.nodes):
	# 			if node.params["detection"][t]:
	# 				node_t.append([node.params["xMin"][t], node.params["yMin"][t], node.params["xMax"][t],
	# 							   node.params["yMax"][t]])
	# 				node_ind.append(ind)
	#
	# 		# Next consider all handshake boxes at time t
	# 		for bbox in handshake_data[str(t)]:
	# 			bb_hs = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
	# 			idx = bbox["id"]
	#
	# 			# iou between bb_hs and bb_person (node_t)
	# 			iou = []
	# 			for i in range(len(node_t)):
	# 				try:
	# 					iou.append(get_iou(bb_hs, node_t[i], mode=1))
	# 				except Exception as e:
	# 					print(e)
	# 					print(t, i)
	# 					input("Enter something")
	#
	# 			# iou = list(map(lambda x: get_iou(bb_hs, x, mode=1), node_t))
	# 			shakes[idx][t] = iou
	#
	# 	unclassified = shakes.pop(-1)	# non-id shakes
	#
	# 	print(shakes)
	#
	# 	for idx in shakes:
	# 		shake_t = shakes[idx].keys()
	# 		shake_iou = list(shakes[idx].values())
	#
	# 		shakes_iou_avg = np.mean(np.array(shake_iou), axis=0).astype(float)
	#
	# 		# print(shakes_iou_avg)
	#
	# 		p1, p2 = np.argpartition(shakes_iou_avg, -2)[-2:]
	# 		p1, p2 = int(p1), int(p2)
	#
	# 		for t in shake_t:
	# 			self.graph.nodes[p1].params["handshake"][t] = {"person": p2, "confidence": None, "iou": shakes_iou_avg[p1]}
	# 			self.graph.nodes[p2].params["handshake"][t] = {"person": p1, "confidence": None, "iou": shakes_iou_avg[p2]}
	#
	# 	print("Updated the graph")
	#
	# else:
	# 	for t in handshake_frames:
	# 		# First take all the detected nodes at time t
	# 		node_t = []
	# 		node_ind = []
	# 		for ind, node in enumerate(self.graph.nodes):
	# 			if node.params["detection"][t]:
	# 				node_t.append([node.params["xMin"][t], node.params["yMin"][t], node.params["xMax"][t],
	# 							   node.params["yMax"][t]])
	# 				node_ind.append(ind)
	#
	# 		# Next consider all handshake boxes at time t
	# 		# nbox = handshake_data[str(t)]["No of boxes"]
	#
	# 		# print(t, node_t)
	#
	# 		for bbox in handshake_data[str(t)]["bboxes"]:
	# 			bb_hs = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
	# 			conf = bbox["conf"]
	#
	# 			# iou between bb_hs and bb_person (node_t)
	#
	# 			iou = []
	# 			for i in range(len(node_t)):
	# 				try:
	# 					iou.append(get_iou(bb_hs, node_t[i], mode=1))
	# 				except Exception as e:
	# 					print(e)
	# 					print(t, i)
	# 					input("Enter something")
	#
	# 			# iou = list(map(lambda x: get_iou(bb_hs, x, mode=1), node_t))
	#
	# 			# get 2 max values
	# 			ind1, ind2 = np.argpartition(iou, -2)[-2:]
	#
	# 			p1, p2 = node_ind[ind1], node_ind[ind2]
	#
	# 			# print(t, p1, p2, iou)
	#
	# 			self.graph.nodes[p1].params["handshake"][t] = {"person": p2, "confidence": conf, "iou": iou[ind1]}
	# 			self.graph.nodes[p2].params["handshake"][t] = {"person": p1, "confidence": conf, "iou": iou[ind2]}
	#
	# 	print("Updated the graph")

	# def runForBatch(self):
		# self.init_from_json()
		# self.update_handshake()

	def visualize(self, img_handle, WAIT=0):
		img_handle.open()

		for t in range(self.time_series_length):
			# print(t)
			rgb = img_handle.read_frame(t)

			NNHandler_openpose.plot(rgb, self.json_data[str(t)], self.is_tracked)

			cv2.imshow("plot", rgb)
			k = cv2.waitKey(WAIT)
			if k & 0xff == ord('q'): break


		img_handle.close()



if __name__ == "__main__":
	g = Graph()
	g.init_from_json('./data/vid-01-graph.json')

	img_handle = NNHandler_image(format="avi", img_loc='./suren/temp/seq18.avi')

	with_tracker = True

	if with_tracker:
		if os.path.exists('./data/vid-01-openpose_track.json'):
			nn_handle = NNHandler_openpose(openpose_file='./data/vid-01-openpose_track.json', is_tracked=True)
			nn_handle.init_from_json()
			nn_handle.visualize(img_handle)
		elif os.path.exists('./data/vid-01-openpose.json'):
			nn_handle = NNHandler_openpose()
			nn_handle.init_from_json(openpose_file='./data/vid-01-openpose.json')
			nn_handle.create_tracker(img_handle=None)
			nn_handle.save_json('./data/vid-01-openpose_track.json')
		else:
			# TODO : Load from graph
			raise NotImplementedError


	#
	# # To create new json for YOLO HS bbox with tracker
	# try:
	# except:
	# 	img_handle = NNHandler_image(format="avi", img_loc="./suren/temp/seq18.avi")
	# 	img_handle.runForBatch()
	#
	# 	nn_handle.create_tracker(img_handle)
	# 	nn_handle.save_json()
	#
	#
	# # init graph from json
	# try:
	# 	g.init_from_json('./data/vid-01-graph_handshake_track.json')
	# except:
	# 	nn_handle.connectToGraph(g)
	# 	nn_handle.runForBatch()
	#
	# 	print("Created graph with nodes = %d for frames = %d. Param example:" % (g.n_nodes, g.time_series_length))
	# 	for p in g.nodes[0].params:
	# 		print(p, g.nodes[0].params[p])
	#
	# 	g.saveToFile('./data/vid-01-graph_handshake_track.json')

	else:
		nn_handle = NNHandler_openpose(openpose_dir='./suren/temp/output_seq18', is_tracked=False)

		try:
			nn_handle.init_from_json()

			if nn_handle.openpose_file is None:
				nn_handle.save_json('./data/vid-01-openpose.json')
		except:
			raise NotImplementedError  # TODO: Find HS bbox from video

	# init graph from json
	# try:
	# 	g.init_from_json('./data/vid-01-graph_handshake.json')
	# 	print("Loaded graph from : ./data/vid-01-graph_handshake.json")
	# except:
	# 	nn_handle.connectToGraph(g)
	# 	nn_handle.runForBatch()
	#
	# 	print("Created graph with nodes = %d for frames = %d. Param example:" % (g.n_nodes, g.time_series_length))
	# 	for p in g.nodes[0].params:
	# 		print(p, g.nodes[0].params[p])
	#
	# 	g.saveToFile('./data/vid-01-graph_handshake.json')

# g.plot()
