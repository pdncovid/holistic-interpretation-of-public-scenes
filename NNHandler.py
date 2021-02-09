from abc import ABC, abstractmethod 

class NNHandler(ABC):
	def __int__(self):
		pass
	def setInputBlockSize(self,N):
		self.inputBlockSize=N
	
	def getInputBlockSize(self,):
		return self.inputBlockSize

	def connectToInput(self,inputHandlerInstance):
		self.myId = inputHandlerInstance.connectComponent(self.inputBlockSize)

	def connectToGraph(self,gr):
		self.graph=gr

	def updateGraph():
		pass


