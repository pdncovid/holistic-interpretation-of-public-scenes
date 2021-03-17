from abc import ABC, abstractmethod 
from Node import *
class Person(Node):

	def __init__(self,isLocation2D=True,timeSeriesLength=1000):
		self.location2D=isLocation2D
		self.timeSeriesLength=timeSeriesLength
		self.params={}

		#>>>>>>>>>> REMOVE
		self.params["X"]=[0 for _ in range(self.timeSeriesLength)]
		self.params["Y"]=[0 for _ in range(self.timeSeriesLength)]
		#<<<<<<<<<<<<<<<<<<<<


		'''
			Now we work from the varibles below this line only.

		'''
		self.params["xMin"]=[0 for _ in range(self.timeSeriesLength)]
		self.params["yMin"]=[0 for _ in range(self.timeSeriesLength)]
		self.params["xMax"]=[0 for _ in range(self.timeSeriesLength)]
		self.params["yMax"]=[0 for _ in range(self.timeSeriesLength)]

		self.params["detection"]=[False for _ in range(self.timeSeriesLength)]		

		'''
			X and Y are the center of the feet of the person in IMAGE PIXELS
		'''


	def setInitialLocation(self,X,Y,Z=None):
		self.params["X"][0]=X
		self.params["Y"][0]=Y
	
	def addNewParam(self,paramName):
		self.params[paramName]=[0 for _ in range(self.timeSeriesLength)]

	def setParam(self,paramName,t,val):
		self.params[paramName][t]=val

	def getParam(self,paramName,t):
		return self.params[paramName][t]

	def getParamsDict(self):
		return self.params

	def setParamsFromDict(self,dictt):
		self.params=dictt




	
