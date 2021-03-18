from Node import *

#HUH?
class Node_MaskWearingPercentage(Node):
	def init(self, time_series_length=100):
		super().__init__(time_series_length=time_series_length)

	# 	self.params["maskWearingCount"]=[0 for _ in range(timeSeriesLength)]
	# 	self.params["Y"]=[0 for _ in range(timeSeriesLength)]
	# 	graph=self.initParams["graph"]
	# 	graph=self.initParams["group"]
	# 	self.params["graph"]=graph
	# 	self.params["group"]=group
	#
	# def refresh():
	# 	groupSize=len(self.params["group"])
	# 	for t in range(self.timeSeriesLength):
	# 		x=0.0
	# 		y=0.0
	# 		for n in groupSize:
	# 			x+=self.params["graph"]\
	# 			.getNode(self.params["group"][n]).params["X"][t]
	# 			y+=self.params["graph"]\
	# 			.getNode(self.params["group"][n]).params["Y"][t]
	# 		x=x/groupSize
	# 		y=y/groupSize
	# 		self.params["X"][t]=x
	# 		self.params["Y"][t]=y