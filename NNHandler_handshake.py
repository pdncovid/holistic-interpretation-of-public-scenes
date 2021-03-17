from NNHandler import *


class NNHandler_handshake:
	def __int__(self,yoloHandshakeOutput=yoloHandshakeOutput.txt):
		print("Creating an Openpose handler")
		self.inputFileName=yoloHandshakeOutput


	def runForBatch(self):
		
		# print("Getting frames from InputHandler")
		# frames=self.myInput.getFrameBlock(self.myId)
		
		f=open(self.inputFileName,"r")
		#Get the handshake bounding boxes.

		#Use self.graph and find the two people using maximum intersection area

		'''
			SUREN ;-)
		'''


		# self.graph
		n=self.graph.getNode(0)
		n.addParam("handshake")
		for a in range(self.graph.timeSeriesLength):
			n.setParam("handshake",{"person":None,"confidence":None})





		'''
		Put ur code here.
		'''

		print("Updated the graph")
		


	# def processBatch(self,fr):
	# 	print("NN in action")