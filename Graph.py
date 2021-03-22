import json
from Node_Person import Person
from suren.util import  eprint, stop, progress
import numpy as np

try:
	import networkx as nx
	import matplotlib.pyplot as plt
	import matplotlib.cm as cm

except ImportError as e:
	print(e)
	# SHOW = False  # No idea if this would work when importing @all...maybe call as function?


# Graph visualization packages

class Graph:
	def __init__(self, time_series_length=None):
		"""
		:param timeSeriesLength: Number of frames
		"""
		self.time_series_length = time_series_length

		self.n_nodes = 0
		self.n_person = 0
		self.nodes=[]

		self.BIG_BANG=0			# HUH -_-

	@staticmethod
	def plot_import():
		try:
			import networkx
			import matplotlib.pyplot

			return True
		except ImportError as e:
			print(e)
			# SHOW = False  # No idea if this would work when importing @all...maybe call as function?
			return False


	def __repr__(self):
		return "Graph with %d nodes"%self.n_nodes

	def plot(self, window = 10):

		def get_cmap(window=10, show=False):
			colors = cm.hsv(np.linspace(0, 1, self.n_nodes))
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
				plt.scatter(x.flatten(), y.flatten(), color = np.reshape(cmap, (-1, 4), order='F'))
				plt.show()

			return cmap

		if not Graph.plot_import():
			eprint("Network package not installed")
			return

		# plt.figure()

		cmap = get_cmap(window, show=True)


		# G = nx.Graph()
		sc_x = []
		sc_y = []

		for t in range(self.time_series_length):
			pos = {}
			sc_tx, sc_ty = [], []
			for n, p in enumerate(self.nodes):
				# print(n, p.params)
				# if p.params["detection"][t]:
				p_x = p.params["X"][t]
				p_y = p.params["Y"][t]
				pos[n] = (p_x, p_y)
				sc_tx.append(p_x)
				sc_ty.append(p_y)

			sc_x.append(sc_tx)
			sc_y.append(sc_ty)

		sc_x = np.array(sc_x).transpose()
		sc_y = np.array(sc_y).transpose()

		print(sc_x.shape, sc_y.shape, cmap.shape)

		# PLOT
		# @suren... make this interactive (while playing video)

		plt.figure()
		ax = plt.gca()
		# plt.xlim((np.min(sc_x, axis=None))

		plt.ion()

		for t in range(self.time_series_length):
			sc_x_ = sc_x[:, max(t+1-window, 0):t+1]
			sc_y_ = sc_y[:, max(t+1-window, 0):t+1]
			cmap_ = cmap[:, max(0, window-(t+1)):, :]

			# print(sc_x_)
			# print(sc_y_)
			# print(cmap_)

			# print(sc_x_.shape, sc_y_.shape, cmap_.shape)

			ax.scatter(sc_x_.flatten(), sc_y_.flatten(), color=np.reshape(cmap_, (-1, 4), order='C'))

			plt.pause(.1)
			ax.clear()

			if (t+1)%20 ==0:
				progress(t+1, self.time_series_length, "drawing graph")

		# plt.show(block=True)








	def get_nxt_id(self):
		return len(self.nodes)

	def add_person(self, p=None):
		p = Person(time_series_length=self.time_series_length, idx=self.get_nxt_id()) if p is None else p

		self.nodes.append(p)
		self.n_person +=1
		self.n_nodes = len(self.nodes)

		return p

	def addNode(self,time):
		print("GRAPH: adding (person) node")
		self.nodes.append(Person())
		return len(self.nodes)-1

	# def addNode2(self,node):
	# 	'''
	# 	Merge this with addNode()
	# 	'''
	# 	print("GRAPH: adding (general) node")
	# 	self.nodes.append(node)
	# 	return len(self.nodes)-1

	def getNode(self,idx):
		return self.nodes[idx]

	def saveToFile(self,fileName="graph.txt"):
		data={}
		data["N"]=len(self.nodes)
		data["nodes"]=[]
		for n in self.nodes:
			data["nodes"].append(n.getParamsDict())
		print(data)
		with open(fileName, 'w') as outfile:
			json.dump(data, outfile)
		print("Finished writing all nodes to {}".format(fileName))

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
			time_series_length = data["NoFrames"]
			assert len(data["nodes"][0]["detection"]) == time_series_length, "No of nodes not equal to N"
		except Exception as e:
			eprint(e)
			time_series_length = len(data["nodes"][0]["detection"])

		if N == 0:
			return

		if self.time_series_length is None: self.time_series_length = time_series_length
		else: assert self.time_series_length == N, "Graph time is not equal to the json file [N]"

		for n in range(N):
			p = self.add_person()
			p.setParamsFromDict(data["nodes"][n])

		print("Finished reading {} modes from {}".format(len(self.nodes),file_name))



	def loadFromFile(self, fileName="graph.txt"):
		# @gihan check why detection goes from False to 00000
		with open(fileName) as json_file:
			data = json.load(json_file)
		# print("Finished reading")
		N=data["N"]
		for n in range(N):
			p=Person()
			p.setParamsFromDict(data["nodes"][n])
			self.nodes.append(p)

		print("Finished reading {} modes from {}".format(len(self.nodes),fileName))

if __name__ == "__main__":
	g = Graph()
	g.init_from_json('./nn-outputs/sample-YOLO-bbox.json')

	print("Created graph with %s nodes. Param example:"%g.n_nodes)
	print(g.nodes[0].params)

	g.plot()
