import json
import cv2
import numpy as np
from collections import defaultdict

from Node_Person import Person
from suren.util import eprint, stop, progress, Json
# from sklearn.cluster import SpectralClustering

try:
	# import networkx as nx
	import matplotlib.pyplot as plt
	import matplotlib.cm as cm
	from mpl_toolkits.axes_grid1 import make_axes_locatable

except ImportError as e:
	eprint(e)


# SHOW = False  # No idea if this would work when importing @all...maybe call as function?


# Graph visualization packages

class Graph:
	@staticmethod
	def plot_import():
		try:
			# import networkx
			import matplotlib.pyplot as plt
			import matplotlib.cm as cm
			from mpl_toolkits.axes_grid1 import make_axes_locatable
			return None
		except ImportError as e:
			print(e)
			# SHOW = False  # No idea if this would work when importing @all...maybe call as function?
			return e

	def __init__(self, time_series_length=None, save_name=None):
		"""
		:param timeSeriesLength: Number of frames
		"""
		self.time_series_length = time_series_length
		# self.time_series_length = 10
		'''
			Gihan hardcoded to 100 for debug purposes.
		'''

		self.n_nodes = 0
		self.n_person = 0
		self.nodes = []			# arr with n_nodes number of Person Objects

		self.state = {
			"people" : 0, 		# 1 - people bbox, 2 - tracking id
			"handshake" : 0, 	# 1 - hs only, 2 - with tracking id, 3 - person info
			"cluster" : 0,		# 1 - has cluster info
			"mask" : 0,
			"floor" : 0,		# 0b001 - X, Y points only, 0b010 - Interpolation, 0b100 - Projected Floor maps generated with X,Y
			"threat" : 0
		}

		self.saveGraphFileName = save_name

		self.BIG_BANG = 0  # HUH -_-
		self.threatLevel = None

		self.PROJECTED_SPACE_H=1000
		self.PROJECTED_SPACE_W=1000
		self.DEST=[[0,self.PROJECTED_SPACE_H],[self.PROJECTED_SPACE_W,self.PROJECTED_SPACE_H],[self.PROJECTED_SPACE_W,0],[0,0]]

		self.projectedFloorMapNTXY = None
		self.REFERENCE_POINTS = None




	def __repr__(self):
		rep = "Created Graph object with nodes = %d for frames = %d. Param example:\n" % (
		self.n_nodes, self.time_series_length)
		for p in self.nodes[0].params:
			rep += "\t" + str(p) + " " + str(self.nodes[0].params[p]) + "\n"
		return rep

	def project(self,x,y):
		projected=np.dot(self.transMatrix, np.array([x,y,1]))
		projected=[projected[0]/projected[2],projected[1]/projected[2]]
		return projected

	def get_plot_lim(self, sc_x=None, sc_y=None):
		if sc_x is not None and sc_y is not None:
			y_lim = [np.min(sc_y, axis=None) - 5, np.max(sc_y, axis=None) + 5]
			x_lim = [np.min(sc_x, axis=None) - 5, np.max(sc_x, axis=None) + 5]

		else:

			# @GIHAN. TODO : Put 4 endpoints here
			x_min = 0
			x_max = 0
			y_min = 0
			y_max = 0
			x_lim = [x_min, x_max]
			y_lim = [y_min, y_max]

			raise NotImplementedError


		return x_lim,y_lim


	def get_plot_points(self):

		assert self.state["floor"] >= 1, "Need X, Y points to plot graph"

		sc_x = []
		sc_y = []
		lines = []
		for t in range(self.time_series_length):
			# pos = {}
			sc_tx, sc_ty = [], []
			line_t = defaultdict(list)
			for n, p in enumerate(self.nodes):
				p_x = p.params["X"][t]
				p_y = p.params["Y"][t]
				p_x, p_y = self.project(p_x, p_y)

				# pos[n] = (p_x, p_y)
				sc_tx.append(p_x)
				sc_ty.append(p_y)

				if p.params["handshake"][t]['person'] is not None:
					n1, n2 = sorted([n, p.params["handshake"][t]['person']])
					line_t["%d_%d" % (n1, n2)].append([p_x, p_y])
					# print(t, n1, n2, p_x, p_y)

			sc_x.append(sc_tx)
			sc_y.append(sc_ty)
			# print("XXX", line_t)
			# @suren : find a better way to implement variable size array
			try:
				line_t = np.array([line_t[l] for l in line_t]).transpose((0, 2, 1))
			except ValueError:
				line_t = []
			lines.append(line_t)
		sc_x = np.array(sc_x).transpose()
		sc_y = np.array(sc_y).transpose()

		return sc_x, sc_y, lines

	def get_cmap(self, n : int = None, show=False):
		if n is None: n = self.n_nodes

		colors = cm.hsv(np.linspace(0, .8, n))
		window = 10
		col_arr = np.ones((window, 4))
		col_arr[:, -1] = np.power(.8, np.arange(window))[::-1]
		arr1 = np.tile(colors, (window, 1, 1)).transpose((1, 0, 2))
		# print(colors.shape, arr1.shape)
		arr2 = np.tile(col_arr, (n, 1, 1))
		# print(col_arr.shape, arr2.shape)
		cmap = arr1 * arr2
		# print(arr1[1, :, :], arr2[1, :, :])
		# print(colors)
		# stop()
		if show:
			x = np.tile(np.arange(cmap.shape[0]), (cmap.shape[1], 1))
			y = np.tile(np.arange(cmap.shape[1]), (cmap.shape[0], 1)).transpose()
			# print(x)
			# print(y)
			plt.figure()
			plt.title("Colour map (Close to continue)")
			plt.scatter(x.flatten(), y.flatten(), color=np.reshape(cmap, (-1, 4), order='F'))
			plt.show()
		return cmap

	# Plot with moving points... don't use this. Visualizer has a better implementation.
	def plot(self, window=10, show_cmap=True):
		if Graph.plot_import() is not None:
			eprint("Package not installed", Graph.plot_import())
			return
		# plt.figure()
		# colour map
		cmap = self.get_cmap(show=show_cmap)
		# scatter x, y and lines
		sc_x, sc_y, lines = self.get_plot_points()
		# print(sc_x.shape, sc_y.shape, cmap.shape)
		# PLOT
		xlim, ylim = self.get_plot_lim(sc_x, sc_y)

		fig = plt.figure()
		plt.xlim(xlim[0], xlim[1])
		plt.ylim(ylim[0], ylim[1])

		ax = plt.gca()
		plt.ion()
		for t in range(self.time_series_length):
			sc_x_ = sc_x[:, max(t + 1 - window, 0):t + 1]
			sc_y_ = sc_y[:, max(t + 1 - window, 0):t + 1]
			cmap_ = cmap[:, max(0, window - (t + 1)):, :]

			# print(sc_x_.shape, sc_y_.shape, cmap_.shape)

			ax.scatter(sc_x_.flatten(), sc_y_.flatten(), color=np.reshape(cmap_, (-1, 4), order='C'))

			for l in lines[t]:
				ax.plot(l[0], l[1])
				plt.pause(.3)


			else:
				plt.pause(.1)

			ax.clear()
			ax.set_xlim(xlim[0], xlim[1])
			ax.set_ylim(ylim[0], ylim[1])

			if (t + 1) % 20 == 0:
				progress(t + 1, self.time_series_length, "drawing graph")

	# plt.show(block=True)

	def get_nxt_id(self):
		return len(self.nodes)

	def add_person(self, p : Person = None):
		if p is None:
			p = Person(time_series_length=self.time_series_length, idx=self.get_nxt_id())
		elif p.idx is None:
			p.idx = self.get_nxt_id()

		self.nodes.append(p)
		self.n_person += 1
		self.n_nodes = len(self.nodes)

		return p

	# def addNode(self,time):
	# 	print("GRAPH: adding (person) node")
	# 	self.nodes.append(Person())
	# 	return len(self.nodes)-1

	def getNode(self, idx):
		return self.nodes[idx]

	def make_jsonable(self, data):
		for node in data["nodes"]:
			for param in node:
				print(param, node[param])
				if param == "handshake":
					for t in range(self.time_series_length):
						print(type(node[param][t]['person']))
						print(type(node[param][t]['confidence']))

				else:
					for t in range(self.time_series_length):
						print(type(node[param][t]))
			# print(type(node["handshake"]))

	def saveToFile(self, file_name=None):
		if file_name is None: file_name = self.saveGraphFileName

		data = {
			"N": self.n_nodes,
			"frames": self.time_series_length,
			"state": self.state,
			"nodes": [n.params for n in self.nodes]
		}

		js = Json(file_name)
		js.write(data)

		print("Finished writing all nodes to {}".format(file_name))

	def getCameraInfoFromJson(self,fileName):
		with open(fileName) as json_file:
			data = json.load(json_file)

		self.REFERENCE_POINTS=data["reference_points"]
		self.REFERENCE_POINTS=np.float32(self.REFERENCE_POINTS)

		self.DEST=np.float32(self.DEST)
		self.transMatrix= cv2.getPerspectiveTransform(self.REFERENCE_POINTS,self.DEST)

		self.GROUP_DIST_THRESH=data["group_radius_threshold"]
		self.GROUP_TIME_THRESH=data["group_time_threshold"]

		self.DISTANCE_TAU = data["distance_tau"]


	def init_from_json(self, file_name):
		with open(file_name) as json_file:
			data = json.load(json_file)

		try:
			N = data["N"]
			assert len(data["nodes"]) == N, "No of nodes not equal to N"
		except Exception as e:
			eprint(e)
			N = len(data["nodes"])

		try:
			time_series_length = data["frames"]
			assert len(data["nodes"][0]["detection"]) == time_series_length, "Time series length not equal"
		except Exception as e:
			eprint(e)
			time_series_length = len(data["nodes"][0]["detection"])

		if N == 0:
			eprint("No nodes :(")
			return

		if self.time_series_length is None: self.time_series_length = time_series_length

		for n in range(N):
			p = self.add_person()
			p.setParamsFromDict(data["nodes"][n])

	# def calculate_standing_locations(self):
	# 	for n in self.nodes:
	# 		n.calculate_standing_locations()

	# def interpolate_undetected_timestamps(self):
	# 	for n in self.nodes:
	# 		n.interpolate_undetected_timestamps()

	def generateFloorMap(self, verbose=False, debug=False):

		assert self.state["people"] >= 2, "Floor map cannot be generated without people bbox"

		if self.state["floor"] < 1:
			for n in self.nodes:
				n.calculate_standing_locations()
			self.state["floor"] = 1

		if self.state["floor"] & 1 << 1 == 0:
			for n in self.nodes:
				n.interpolate_undetected_timestamps(debug=debug)
			self.state["floor"] |= 1 << 1

		# Floor map N x T with X and Y points.
		self.projectedFloorMapNTXY = np.zeros((self.n_nodes, self.time_series_length, 2),dtype=np.float32)

		for n in range(self.n_nodes):
			X = self.nodes[n].params["X"]
			Y = self.nodes[n].params["Y"]
			for t in range(self.time_series_length):

				projected=self.project(X[t], Y[t])
				self.projectedFloorMapNTXY[n,t,0]=projected[0]
				self.projectedFloorMapNTXY[n,t,1]=projected[1]

		self.state["floor"] |= 1 << 2

		if verbose:
			print("Finished creating floormap {} ".format(self.projectedFloorMapNTXY.shape))


	def findClusters(self, METHOD="NAIVE", debug=False, verbose=False):

		self.groupProbability = np.zeros((self.n_nodes, self.n_nodes, self.time_series_length), np.float32)

		#There is a lot for me to do on this array.
		self.pairDetectionProbability = np.zeros((self.n_nodes, self.n_nodes, self.time_series_length), np.float32)

		if METHOD=="NAIVE":
			for p1 in range(self.n_nodes):
				for p2 in range(self.n_nodes):
					if p1<=p2:
						if not self.nodes[p1].params["neverDetected"] and not self.nodes[p2].params["neverDetected"]:
							t1=max(self.nodes[p1].params["detectionStartT"],self.nodes[p2].params["detectionStartT"])
							t2=max(self.nodes[p1].params["detectionEndTExclusive"],self.nodes[p2].params["detectionEndTExclusive"])
							self.pairDetectionProbability[p1,p2,t1:t2]=1.00
						

						tempDistDeleteThisVariableLater=[]
						for t in range(self.time_series_length):
							dist=np.sqrt(np.sum(np.power(self.projectedFloorMapNTXY[p1,t,:]-self.projectedFloorMapNTXY[p2,t,:],2)))
							tempDistDeleteThisVariableLater.append(dist)
							if dist<self.GROUP_DIST_THRESH:
								self.groupProbability[p1,p2,t]=1.0
							else:
								self.groupProbability[p1,p2,t]=0.0

						if debug:
							# print("Dist between {} and  {}:".format(p1,p2),tempDistDeleteThisVariableLater)
							print("Dist between {} and  {}:".format(p1,p2),tempDistDeleteThisVariableLater[t1:t2])
					else:
						self.groupProbability[p1,p2,:]=self.groupProbability[p2,p1,:]
						self.pairDetectionProbability[p1,p2,:]=self.pairDetectionProbability[p2,p1,:]




		if METHOD=="SPECTRAL":
			print("THIS METHOD WAS REMOVED!!!")
			""" <<<< end """


		# (a,b,c) are temporary variables
		a = self.groupProbability*self.pairDetectionProbability
		b = np.sum(a,-1)
		c = np.sum(self.pairDetectionProbability,-1)
		self.groupProbability = b/c

		if verbose:
			print("Group probability", self.groupProbability)

		self.groupProbability = self.groupProbability > self.GROUP_TIME_THRESH

		if verbose:
			print("Group probability (bianry)",self.groupProbability)

		self.state["cluster"] = 1


	def calculateThreatLevel(self, debug=False):
		P=len(self.nodes)
		T=self.time_series_length


		self.pairD=np.zeros((T,P,P),dtype=np.float32)
		self.pairI=np.zeros((T,P,P),dtype=np.float32)
		self.pairM=np.zeros((T,P,P),dtype=np.float32)
		self.pairG=np.zeros((T,P,P),dtype=np.float32)
		self.pairT=np.zeros((T,P,P),dtype=np.float32)
		self.frameThreatLevel=np.zeros((T),dtype=np.float32)

		for t in range(T):
			threatLevel = 0.0


			for p1 in range(P):
				interact = self.nodes[p1].params["handshake"][t]

				for p2 in range(P):
					if p1 != p2:
						d=np.linalg.norm(self.projectedFloorMapNTXY[p1,t,:]-self.projectedFloorMapNTXY[p2,t,:])
						d = np.exp(-1.0*d/self.DISTANCE_TAU)
						i = 1 if interact["person"] == p2 else 0 #get from graph self.nodes @Jameel
						m = 0.0 #get from graph self.nodes @Suren
						g = self.groupProbability[p1,p2]
						self.pairD[t,p1,p2]=d
						self.pairI[t,p1,p2]=i
						self.pairM[t,p1,p2]=m
						self.pairG[t,p1,p2]=g

						EPS_m = 2.0
						EPS_g = 2.0
						threatOfPair = (d+i)*(EPS_m-m)*(EPS_g-g)
						threatLevel += threatOfPair

						self.pairT[t,p1,p2]=threatOfPair
			self.frameThreatLevel[t]=threatLevel
		print("Finished calculating threat level")

		self.state["threat"] = 1
		return 0

	def fullyAnalyzeGraph(self):
		self.generateFloorMap()
		self.findClusters()
		self.calculateThreatLevel()

	def gihan_init(self, fig, ax):
		vals = {
			"d" : (self.pairD[0, :, :], "Distance"),
			"i" : (self.pairI[0, :, :], "Interaction"),
			"m" : (self.pairM[0, :, :], "Mask"),
			"g" : (self.pairG[0, :, :], "Group")
		}

		if len(ax) == 2:
			ax = [ax[i][j] for i in range(2) for j in range(2)]

		for col, ind in zip(ax, vals):
			col.title.set_text(vals[ind][1])
			im = col.matshow(vals[ind][0], vmin=0, vmax=1)
			divider = make_axes_locatable(col)
			cax = divider.append_axes('right', size='5%', pad=0.05)
			fig.colorbar(im, cax=cax, orientation='vertical')

	def threat_image(self, fig, out_name, t):
		fig.clf()
		ax = fig.add_axes([0, 0, 1, 1])
		im = ax.matshow(self.pairT[t, :, :])
		divider = make_axes_locatable(ax)
		cax = divider.append_axes('right', size='5%', pad=0.05)
		fig.colorbar(im, cax=cax, orientation='vertical')
		fig.savefig("{}T-{:04d}.jpg".format(out_name, t))

	def gihan_images(self, fig, ax, out_name, t, concat=True):

		vals = {
			"d" : (self.pairD[t, :, :], "Distance"),
			"i" : (self.pairI[t, :, :], "Interaction"),
			"m" : (self.pairM[t, :, :], "Mask"),
			"g" : (self.pairG[t, :, :], "Group")
		}

		if len(ax) == 2:
			ax = [ax[i][j] for i in range(2) for j in range(2)]

		if concat:
			for col, ind in zip(ax, vals):
				col.title.set_text(vals[ind][1])
				col.matshow(vals[ind][0], vmin=0, vmax=1)

			fig.savefig("{}dimg-{:04d}.jpg".format(out_name, t))
			for col in ax:
				col.clear()


		else:
			raise NotImplementedError



		# plt.figure()
		# plt.matshow(self.pairI[t, :, :], vmin=0, vmax=1)
		# plt.colorbar()
		# plt.savefig("{}i-{:04d}".format(out_name, t))
		# plt.close()



if __name__ == "__main__":
	g = Graph()
	# g.init_from_json('./data/vid-01-graph.json')		# Start from yolo
	
	g.init_from_json('./data/vid-01-graph_handshake.json')  # Start from handshake
	g.getCameraInfoFromJson('./data/camera-orientation/jsons/deee.json')
	g.fullyAnalyzeGraph()

	# print("Created graph with nodes = %d for frames = %d. Param example:" % (g.n_nodes, g.time_series_length))
	print(g.pairD)



