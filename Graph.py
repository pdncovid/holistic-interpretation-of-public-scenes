import json
import cv2
import numpy as np
from collections import defaultdict

from Node_Person import Person
from suren.util import eprint, stop, progress, Json
from sklearn.cluster import SpectralClustering

try:
	# import networkx as nx
	import matplotlib.pyplot as plt
	import matplotlib.cm as cm

except ImportError as e:
	print(e)


# SHOW = False  # No idea if this would work when importing @all...maybe call as function?


# Graph visualization packages

class Graph:
	def __init__(self, time_series_length=None, save_name=None):
		"""
		:param timeSeriesLength: Number of frames
		"""
		self.time_series_length = time_series_length

		self.n_nodes = 0
		self.n_person = 0
		self.nodes = []
		self.saveGraphFileName = save_name

		self.BIG_BANG = 0  # HUH -_-

		self.PROJECTED_SPACE_H=1000
		self.PROJECTED_SPACE_W=1000

		self.REFERENCE_POINTS=[[60,1080],[1850,1080],[1450,450],[920,450]]
		self.DEST=[[0,self.PROJECTED_SPACE_H],[self.PROJECTED_SPACE_W,self.PROJECTED_SPACE_H],[self.PROJECTED_SPACE_W,0],[0,0]]
		self.REFERENCE_POINTS=np.float32(self.REFERENCE_POINTS)
        self.DEST=np.float32(self.DEST)
        self.transMatrix=cv.getPerspectiveTransform(self.REFERENCE_POINTS,self.DEST)
	
	def project(self,x,y):
		projected=np.dot(self.transMatrix, np.array([x,y,1]))
		projected=[projected[0]/projected[2],projected[1]/projected[2]]
		return projected

	@staticmethod
	def plot_import():
		try:
			# import networkx
			import matplotlib.pyplot

			return None
		except ImportError as e:
			print(e)
			# SHOW = False  # No idea if this would work when importing @all...maybe call as function?
			return e

	def __repr__(self):
		rep = "Created Graph object with nodes = %d for frames = %d. Param example:\n" % (self.n_nodes, self.time_series_length)
		for p in self.nodes[0].params:
			rep += "\t" + str(p) + " " + str(self.nodes[0].params[p]) + "\n"

		return rep

	def get_plot_points(self):
		sc_x = []
		sc_y = []
		lines = []

		for t in range(self.time_series_length):
			# pos = {}
			sc_tx, sc_ty = [], []
			line_t = defaultdict(list)
			for n, p in enumerate(self.nodes):
				# print(n, p.params)
				# if p.params["detection"][t]:
				p_x1 = p.params["xMin"][t]
				p_y1 = p.params["yMin"][t]
				p_x2 = p.params["xMax"][t]
				p_y2 = p.params["yMax"][t]
				p_x = (p_x1 + p_x2) / 2
				p_y = (p_y1 + p_y2) / 2
				# pos[n] = (p_x, p_y)
				sc_tx.append(p_x)
				sc_ty.append(p_y)

				if p.params["handshake"][t]['person'] is not None:
					n1, n2 = sorted([n, p.params["handshake"][t]['person']])
					line_t["%d_%d"%(n1, n2)].append([p_x, p_y])
					# print(t, n1, n2, p_x, p_y)

			sc_x.append(sc_tx)
			sc_y.append(sc_ty)

			# print("XXX", line_t)

			# @suren : find a better way to implement variable size array
			try: line_t = np.array([line_t[l] for l in line_t]).transpose((0, 2, 1))
			except ValueError: line_t = []


			lines.append(line_t)

		sc_x = np.array(sc_x).transpose()
		sc_y = np.array(sc_y).transpose()
		# lines = np.array(lines)			# lines is a variable size array

		return sc_x, sc_y, lines

	def get_cmap(self, show=False):
		colors = cm.hsv(np.linspace(0, .8, self.n_nodes))
		window = 10

		col_arr = np.ones((window, 4))

		col_arr[:, -1] = np.power(.8, np.arange(window))[::-1]

		arr1 = np.tile(colors, (window, 1, 1)).transpose((1, 0, 2))
		# print(colors.shape, arr1.shape)
		arr2 = np.tile(col_arr, (self.n_nodes, 1, 1))
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

	def plot(self, window=10, show_cmap=True):

		if Graph.plot_import() is not None:
			eprint("Package not installed", Graph.plot_import())
			return

		# plt.figure()

		cmap = self.get_cmap(show=show_cmap)

		sc_x, sc_y, lines = self.get_plot_points()

		# print(sc_x.shape, sc_y.shape, cmap.shape)

		# PLOT
		ylim = [np.min(sc_y, axis=None)-5, np.max(sc_y, axis=None)+5]
		xlim = [np.min(sc_x, axis=None)-5, np.max(sc_x, axis=None)+5]

		fig = plt.figure()
		plt.xlim(xlim[0], xlim[1])
		plt.ylim(ylim[0], ylim[1])
		ax = plt.gca()
		# plt.xlim((np.min(sc_x, axis=None))

		plt.ion()

		for t in range(self.time_series_length):
			sc_x_ = sc_x[:, max(t + 1 - window, 0):t + 1]
			sc_y_ = sc_y[:, max(t + 1 - window, 0):t + 1]
			cmap_ = cmap[:, max(0, window - (t + 1)):, :]

			# print(sc_x_)
			# print(sc_y_)
			# print(cmap_)

			# print(sc_x_.shape, sc_y_.shape, 

	def generateFloorMap(self):

		self.floorMap=[]
		for n in range(len(self.nodes)):
			X=self.nodes[n].params["X"]
			Y=self.nodes[n].params["Y"]
			temp=[]
			for t in range(len(X)):
				temp.append(self.project(X[t],Y[t]))
			self.floorMap.append(temp)

	def findClusters(self):
		N=len(self.nodes)
		T=self.time_series_length
		self.groupProbability=np.zeros((N,N,self.time_series_length),np.float)
		clusters=[]
		for t in range(T):
			clusteringAtT = SpectralClustering(n_clusters=2,assign_labels='discretize',random_state=0).fit(self.floorMap[:,t])
			cc=np,max(clusteringAtT.labels)
			#Look at the labels and populate the matrix.
			

		self.groupProbability=np,mean(self.groupProbability,axis=-1)