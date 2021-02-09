class Person:

	def __init__(self,isLocation2D=True,timeSeriesLength=1000):
		self.location2D=isLocation2D
		self.timeSeriesLength=timeSeriesLength
		self.params={}
		self.params["X"]=[0 for _ in range(self.timeSeriesLength)]
		self.params["Y"]=[0 for _ in range(self.timeSeriesLength)]

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




	
