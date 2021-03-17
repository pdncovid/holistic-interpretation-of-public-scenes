from NNHandler import *


class NNHandler_handshake:
	def __int__(self):
		print("Creating an Openpose handler")


	def runForBatch(self):
		
		print("Getting frames from InputHandler")
		frames=self.myInput.getFrameBlock(self.myId)
		
		'''
		Put ur code here.
		'''

		print("Updated the graph")
		# self.graph


	# def processBatch(self,fr):
	# 	print("NN in action")