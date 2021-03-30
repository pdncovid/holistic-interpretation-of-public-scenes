# from abc import ABC, abstractmethod
import json

class NNHandler:
	def __init__(self, graph=None):
		self.graph = graph
		self.myInput = None
		self.myId = None

		self.time_series_length = None
		self.json_data = None

	def setInputBlockSize(self,N=32):
		self.inputBlockSize=N
	
	def getInputBlockSize(self):
		return self.inputBlockSize

	def connectToInput(self,inputHandlerInstance):
		self.myInput = inputHandlerInstance
		self.myId = inputHandlerInstance.connectComponent(self.inputBlockSize)

	def connectToGraph(self,gr):
		self.graph = gr

	def runForBatch(self):
		raise NotImplementedError

	def init_from_json(self, file_name=None):
		# Read json file
		pass
		# raise NotImplementedError




