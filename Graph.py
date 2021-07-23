import json, os
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

		self.pairD = None
		self.pairI = None
		self.pairM = None
		self.pairG = None
		self.pairT = None
		self.frameThreatLevel = None

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

	def get_plot_lim(self, sc_x=None, sc_y=None, hw:tuple=None):

		# LIM based on ref box
		x_min, x_max = np.min(self.DEST[:, 0], axis=None), np.max(self.DEST[:, 0], axis=None)
		y_min, y_max = np.min(-1*self.DEST[:, 1], axis=None), np.max(-1*self.DEST[:, 1], axis=None)

		x_diff = x_max - x_min
		y_diff = y_max - y_min
		r = .05

		x_min -= x_diff * r
		x_max += y_diff * r
		y_min -= y_diff * r
		y_max += y_diff * r

		x_lim = [x_min, x_max]
		y_lim = [y_min, y_max]

		# LIM based on plot points
		if sc_x is not None and sc_y is not None:

			x_min = np.nanmin(sc_x, axis=None)
			x_max = np.nanmax(sc_x, axis=None)

			y_min = np.nanmin(sc_y, axis=None)
			y_max = np.nanmax(sc_y, axis=None)

			x_diff = x_max - x_min
			y_diff = y_max - y_min
			r = .05

			x_min -= x_diff*r
			x_max += y_diff*r

			y_min -= y_diff*r
			y_max += y_diff*r

			x_lim[0] = min(x_lim[0], x_min)
			x_lim[1] = max(x_lim[1], x_max)
			y_lim[0] = min(y_lim[0], y_min)
			y_lim[1] = max(y_lim[1], y_max)

		# LIM based on video size
		if hw is not None:

			raise NotImplementedError

			# @GIHAN. TODO : Put 4 endpoints here
			print(hw)
			# x2, y2 = self.project(hw[1], hw[0])


			x_diff = x_max - x_min
			y_diff = y_max - y_min
			r = .01

			x_min -= x_diff*r
			x_max += y_diff*r

			y_min -= y_diff*r
			y_max += y_diff*r

			x_lim[0] = min(x_lim[0], x_min)
			x_lim[1] = max(x_lim[1], x_max)
			y_lim[0] = min(y_lim[0], y_min)
			y_lim[1] = max(y_lim[1], y_max)

		return x_lim,y_lim

	def get_points_t(self, t):
		scx_det, scy_det, id_det = [], [], []
		scx_interp, scy_interp, id_interp = [], [], []
		line_t = {}
		for n, p in enumerate(self.nodes):
			# initParams = {"id": idx, "xMin": x_min, "xMax": x_max, "yMin": y_min, "yMax": y_max, "detection": detected})
			p_id = p.params["id"]
			# p_x = p.params["X"][t]
			# p_y = p.params["Y"][t]

			if p.params["detection"][t]:
				p_x, p_y = p.params["X_project"][t], p.params["Y_project"][t]

				scx_det.append(p_x)
				scy_det.append(-p_y)
				id_det.append(p_id)

				if p.params["handshake"][t]['person'] is not None:
					n1, n2 = sorted([n, p.params["handshake"][t]['person']])
					line_t["%d_%d" % (n1, n2)].append([p_x, p_y])

			if p.params["interpolated"][t]:
				p_x, p_y = p.params["X_project"][t], p.params["Y_project"][t]

				scx_interp.append(p_x)
				scy_interp.append(-p_y)
				id_interp.append(p_id)


		line_t = np.array([line_t[l] for l in line_t])
		if len(line_t)>0 : line_t = line_t.transpose((0, 2, 1))

		id_det = np.array(id_det, dtype=int)
		id_interp = np.array(id_interp, dtype=int)

		return scx_det, scy_det, id_det, line_t, scx_interp, scy_interp, id_interp


	def get_scatter_points(self):
		sc_x = []
		sc_y = []
		for t in range(self.time_series_length):
			sc_tx, sc_ty = [], []
			for p in self.nodes:
				p_x = p.params["X"][t]
				p_y = p.params["Y"][t]

				if p.params["detection"][t]:
					p_x, p_y = self.project(p_x, p_y)

					# pos[n] = (p_x, p_y)
					sc_tx.append(p_x)
					sc_ty.append(-p_y)
				else:

					sc_tx.append(None)
					sc_ty.append(None)

			sc_x.append(sc_tx)
			sc_y.append(sc_ty)
		sc_x = np.array(sc_x, dtype=float).transpose()
		sc_y = np.array(sc_y, dtype=float).transpose()

		return sc_x, sc_y



	def get_plot_points_all(self):

		# assert self.state["floor"] >= 1, "Need X, Y points to plot graph"     # @suren : TODO

		# ori_x = []
		# ori_y = []

		sc_x = []
		sc_y = []
		lines = []
		for t in range(self.time_series_length):
			# pos = {}
			# ori_tx, ori_ty = [], []
			sc_tx, sc_ty = [], []
			line_t = defaultdict(list)
			for n, p in enumerate(self.nodes):
				p_x = p.params["X"][t]
				p_y = p.params["Y"][t]

				if p.params["detection"][t]:
					# ori_tx.append(p_x)
					# ori_ty.append(p_y)
					p_x, p_y = self.project(p_x, p_y)

					# pos[n] = (p_x, p_y)
					sc_tx.append(p_x)
					sc_ty.append(-p_y)

					if p.params["handshake"][t]['person'] is not None:
						n1, n2 = sorted([n, p.params["handshake"][t]['person']])
						line_t["%d_%d" % (n1, n2)].append([p_x, p_y])
						# print(t, n1, n2, p_x, p_y)
				else:
					# ori_tx.append(None)
					# ori_ty.append(None)

					sc_tx.append(None)
					sc_ty.append(None)

			sc_x.append(sc_tx)
			sc_y.append(sc_ty)

			# ori_x.append(ori_tx)
			# ori_y.append(ori_ty)

			# print("XXX", line_t)
			# @suren : find a better way to implement variable size array
			try:
				line_t = np.array([line_t[l] for l in line_t]).transpose((0, 2, 1))
			except ValueError:
				line_t = []
			lines.append(line_t)
		sc_x = np.array(sc_x, dtype=float).transpose()
		sc_y = np.array(sc_y, dtype=float).transpose()

		# ori_x = np.array(ori_x, dtype=float).transpose()
		# ori_y = np.array(ori_y, dtype=float).transpose()

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

		Json.is_jsonable(data)  # Delete this later @suren

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

		if N == 0:
			eprint("No nodes :(")
			return

		try:
			time_series_length = data["frames"]
			assert len(data["nodes"][0]["detection"]) == time_series_length, "Time series length not equal"
		except Exception as e:
			eprint(e)
			time_series_length = len(data["nodes"][0]["detection"])

		self.state = data["state"]


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
			# self.state["floor"] = 1     # @suren : TODO

		# if True:
			for n in self.nodes:
				n.interpolate_undetected(debug=debug)
			# self.state["floor"] |= 1 << 1     # @suren : TODO

		# if True:
			for n in self.nodes:
				n.project_standing_location(self.transMatrix)



		# Floor map N x T with X and Y points.
		self.projectedFloorMapNTXY = np.zeros((self.n_nodes, self.time_series_length, 2),dtype=np.float32)

		for n, node in enumerate(self.nodes):
			self.projectedFloorMapNTXY[n, :, 0] = node.params["X_project"]
			self.projectedFloorMapNTXY[n, :, 1] = node.params["Y_project"]

			# X = self.nodes[n].params["X"]
			# Y = self.nodes[n].params["Y"]
			# for t in range(self.time_series_length):
			#
			# 	projected=self.project(X[t], Y[t])
			# 	self.projectedFloorMapNTXY[n,t,0]=projected[0]
			# 	self.projectedFloorMapNTXY[n,t,1]=projected[1]
			#
			# np.testing.assert_almost_equal(self.projectedFloorMapNTXY[n, :, 0], node.params["X_project"], decimal = 5)
			# np.testing.assert_almost_equal(self.projectedFloorMapNTXY[n, :, 1], node.params["Y_project"], decimal =5)
			#
			# raise NotImplementedError

		# self.state["floor"] |= 1 << 2    # @suren : TODO
		self.state["floor"] = 1

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

		# self.state["cluster"] = 1     # @suren : TODO


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
						
						self.pairD[t,p1,p2]=d*self.pairDetectionProbability[p1,p2,t]
						self.pairI[t,p1,p2]=i*self.pairDetectionProbability[p1,p2,t]
						self.pairM[t,p1,p2]=m*self.pairDetectionProbability[p1,p2,t]
						self.pairG[t,p1,p2]=g*self.pairDetectionProbability[p1,p2,t]



						EPS_m = 2.0
						EPS_g = 2.0
						threatOfPair = (d+i)*(EPS_m-m)*(EPS_g-g)*self.pairDetectionProbability[p1,p2,t]
						threatLevel += threatOfPair

						self.pairT[t,p1,p2]=threatOfPair
			self.frameThreatLevel[t]=threatLevel
		print("Finished calculating threat level")

		self.state["threat"] = 1     # @suren : TODO

	def fullyAnalyzeGraph(self):
		self.generateFloorMap()
		self.findClusters()
		self.calculateThreatLevel()


	def set_ax(self, ax, n):
		ax.set_xticks(range(n))
		ax.set_yticks(range(n))
		ax.set_xticklabels([str(i+1) for i in range(n)])
		ax.set_yticklabels([str(i+1) for i in range(n)])

	def threat_image_init(self, fig, ax):
		# ax.spines.right.set_visible(False)
		# ax.spines.bottom.set_visible(False)
		# ax.tick_params(bottom=False, labelbottom=False)
		T_max = np.max(self.pairT, axis=None)
		im = ax.matshow(self.pairT[0, :, :], vmin=0, vmax=T_max)
		divider = make_axes_locatable(ax)
		cax = divider.append_axes('right', size='5%', pad=0.05)
		fig.colorbar(im, cax=cax, orientation='vertical')
		ax.clear()
		fig.savefig("./data/output/threat_image_init.jpg")

	def threat_image_save(self, fig, ax, out_name, t):
		T_max = np.max(self.pairT, axis=None)
		ax.matshow(self.pairT[t, :, :], vmin=0, vmax=T_max)
		self.set_ax(ax, self.n_nodes)
		# n, m = self.pairT[t, :, :].shape
		# ax.set_xticklabels([str(i) for i in range(n+1)])
		# ax.set_yticklabels([str(i) for i in range(m+1)])
		fig.savefig("{}T-{:04d}.jpg".format(out_name, t))
		ax.clear()

	def threat_image(self, fig, out_name, t):
		fig.clf()
		ax = fig.add_axes([0, 0, 1, 1])
		im = ax.matshow(self.pairT[t, :, :])
		divider = make_axes_locatable(ax)
		cax = divider.append_axes('right', size='5%', pad=0.05)
		fig.colorbar(im, cax=cax, orientation='vertical')
		fig.savefig("{}T-{:04d}.jpg".format(out_name, t))

	def image_init(self, ax, xlim, ylim):
		ax.set_xlim(xlim[0], xlim[1])
		ax.set_ylim(ylim[0], ylim[1])
		ax.clear()
		ax.set_axis_off()

	def image_save(self, fig1, ax1, xlim, ylim, out_dir, t, clear=True):


		fig1.savefig("{}G-{:04d}.jpg".format(out_dir, t))
		if clear:
			ax1.clear()
			ax1.set_xlim(xlim[0], xlim[1])
			ax1.set_ylim(ylim[0], ylim[1])
			ax1.set_axis_off()

	def dimg_init_concat(self, fig, ax):
		vals = {
			"d" : (self.pairD[0, :, :], "Distance"),
			"i" : (self.pairI[0, :, :], "Interaction"),
			"m" : (self.pairM[0, :, :], "Mask"),
			"g" : (self.pairG[0, :, :], "Group")
		}

		if len(ax) == 2:
			ax = [ax[i][j] for i in range(2) for j in range(2)]

		for col, ind in zip(ax, vals):
			# col.spines.right.set_visible(False)
			# col.spines.top.set_visible(False)
			col.title.set_text(vals[ind][1])
			im = col.matshow(vals[ind][0], vmin=0, vmax=1)
			divider = make_axes_locatable(col)
			cax = divider.append_axes('right', size='5%', pad=0.05)
			fig.colorbar(im, cax=cax, orientation='vertical')

		for col in ax:
			col.clear()
		# 	col.spines.right.set_visible(False)
		# 	col.spines.top.set_visible(False)

		fig.savefig("./data/output/dimg_init_concat.png")

	def dimg_init_full(self, fig, ax):
		ax.clear()
		# ax.spines.right.set_visible(False)
		# ax.spines.top.set_visible(False)
		im = ax.matshow(self.pairD[0, :, :], vmin=0, vmax=1)
		divider = make_axes_locatable(ax)
		cax = divider.append_axes('right', size='5%', pad=0.05)
		fig.colorbar(im, cax=cax, orientation='vertical')
		ax.clear()
		fig.savefig("./data/output/dimg_init_full.jpg")

	def dimg_init(self, fig2, ax2, fig4, ax4):
		self.dimg_init_concat(fig2, ax2)
		self.dimg_init_full(fig4, ax4)

	def dimg_save_concat(self, fig, ax, out_name, t):
		vals = {
			"d" : (self.pairD[t, :, :], "Distance"),
			"i" : (self.pairI[t, :, :], "Interaction"),
			"m" : (self.pairM[t, :, :], "Mask"),
			"gr" : (self.pairG[t, :, :], "Group")
		}

		if len(ax) == 2:
			ax = [ax[i][j] for i in range(2) for j in range(2)]

		n, m = self.pairD[t, :, :].shape

		for col, ind in zip(ax, vals):
			col.title.set_text(vals[ind][1])
			col.matshow(vals[ind][0], vmin=0, vmax=1)
			self.set_ax(col, self.n_nodes)
			# col.set_xticklabels([str(i) for i in range(n+1)])
			# col.set_yticklabels([str(i) for i in range(m+1)])
		fig.savefig("{}dimg-{:04d}.jpg".format(out_name, t))
		for col in ax:
			col.clear()

	def dimg_save_full(self, fig, ax, out_name, t):
		if not os.path.exists("{}/figs".format(out_name)) : os.makedirs("{}/figs".format(out_name))
		vals = {
			"d" : (self.pairD[t, :, :], "Distance"),
			"i" : (self.pairI[t, :, :], "Interaction"),
			"m" : (self.pairM[t, :, :], "Mask"),
			"gr" : (self.pairG[t, :, :], "Group")
		}

		n, m = self.pairD[t, :, :].shape
		for ind in vals:
			ax.matshow(vals[ind][0], vmin=0, vmax=1)
			self.set_ax(ax, self.n_nodes)
			# ax.set_xticklabels([str(i) for i in range(n+1)])
			# ax.set_yticklabels([str(i) for i in range(m+1)])
			fig.savefig("{}/figs/{}-{:04d}.jpg".format(out_name, ind, t))
			ax.clear()

	def dimg_save(self, fig2, ax2, fig4, ax4, out_name, t):
		self.dimg_save_concat(fig2, ax2, out_name, t)
		self.dimg_save_full(fig4, ax4, out_name, t)


if __name__ == "__main__":
	g = Graph()
	# g.init_from_json('./data/vid-01-graph.json')		# Start from yolo
	
	g.init_from_json('./data/vid-01-graph_handshake.json')  # Start from handshake
	g.getCameraInfoFromJson('./data/camera-orientation/jsons/deee.json')
	g.fullyAnalyzeGraph()

	# print("Created graph with nodes = %d for frames = %d. Param example:" % (g.n_nodes, g.time_series_length))
	print(g.pairD)



