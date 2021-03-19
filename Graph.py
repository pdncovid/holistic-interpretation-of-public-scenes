import json
from Person import *

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

	def add_person(self, p=None):
		idx = len(self.nodes)
		p = Person(time_series_length=self.time_series_length, idx=idx) if p is None else p

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

		N = data["N"]

		if self.time_series_length is None: self.time_series_length = N
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

	for p in g.nodes:
		print(p.params)
