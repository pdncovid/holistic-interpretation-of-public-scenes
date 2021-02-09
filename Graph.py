class Graph:
	def __init__(self,timeSeriesLength):
		self.nodes=[]
		self.BIG_BANG=0
		self.TIME_SERIES_LENGTH=timeSeriesLength

	def addNode(self,time):
		self.nodes.append(Person())
		return len(self.nodes)-1


	def getNode(self,idx):
		return self.nodes[idx]


	