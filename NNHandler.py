# from abc import ABC, abstractmethod
import json

class NNHandler:

	def setInputBlockSize(self,N=32):
		self.inputBlockSize=N
	
	def getInputBlockSize(self):
		return self.inputBlockSize

	def connectToInput(self,inputHandlerInstance):
		self.myInput=inputHandlerInstance
		self.myId = inputHandlerInstance.connectComponent(self.inputBlockSize)

	def connectToGraph(self,gr):
		self.graph=gr

	def runForBatch(self):
		raise NotImplementedError

	def init_from_json(self):
		# Read json file
		raise NotImplementedError




