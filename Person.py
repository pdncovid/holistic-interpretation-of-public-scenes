class Person:
	self.location2D=None
	self.locationX=[]
	self.locationY=[]
	self.locationZ=[]
	self.parameters={}

	def __init__(self,isLocation2D=True,timeSeriesLength=1000):
		self.location2D=isLocation2D
		self.timeSeriesLength=timeSeriesLength

	def setInitialLocation(self,X,Y,Z=None):
		self.locationX=X
		self.locationY=Y
		if Z!=None:
			self.locationZ=Z
	
	def addNewParameter(self,paramName):
		self.parameters[paramName]=[]

	def setX(self,t,X):
		self.locationX[t]=X

	def setY(self,t,Y):
		self.locationY[t]=Y





	
