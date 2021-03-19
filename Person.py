from abc import ABC, abstractmethod 
from Node import *
class Person(Node):

	def __init__(self, isLocation2D=True, initParams=None, time_series_length=1000, idx=None):
		"""
		@gihan what are these params
		:param isLocation2D: ??? @gihan
		:param time_series_length: Number of time samples
		"""

		super().__init__(initParams=initParams, time_series_length=time_series_length, idx=idx)

		self.location2D=isLocation2D
		# self.type = 'Person'

		self.init_pos()
		self.init_handshake()



	# def setInitialLocation(self,X,Y,Z=None):
	# 	self.params["X"][0]=X
	# 	self.params["Y"][0]=Y


	def init_handshake(self):
		self.params["handshake"]=[{"person":None,"confidence":None} for _ in range(self.time_series_length)]

		# self.addParam("handshake")
		# for t in range(self.time_series_length):	# @gihan : Why are there multiple time series lengths? Shouldn't this be global?
		# 	self.setParam("handshake", t, {"person":None,"confidence":None})

	def init_pos(self):
		self.params["xMin"]=[0 for _ in range(self.time_series_length)]
		self.params["yMin"]=[0 for _ in range(self.time_series_length)]
		self.params["xMax"]=[0 for _ in range(self.time_series_length)]
		self.params["yMax"]=[0 for _ in range(self.time_series_length)]

		self.params["detection"]=[False for _ in range(self.time_series_length)]






	
