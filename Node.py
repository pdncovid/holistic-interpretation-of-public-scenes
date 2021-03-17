from abc import ABC, abstractmethod 

class Node(ABC):
	def __init__(self,initParams=None,timeSeriesLength=100):
		self.timeSeriesLength=timeSeriesLength
		self.initParams=initParams
		self.params={}
		init()

	def init(self):
		'''
		This function is called for every node after the init.
		'''
		pass

	def setType(self,ty):
		self.type=ty

	def addParam(self,param):
		self.params[param]=[0 for _ in range(self.timeSeriesLength)]
	
	def addStaticParam(self,param,val):
		self.params[param]=val

