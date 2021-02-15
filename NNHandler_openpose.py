from NNHandler import *

class NNHandler_openpose(NNHandler):
	def __int__(self):
		print("Creating an Openpose handler")


	def runForBatch(self):
		
		print("Getting frames from InputHandler")
		frames=self.myInput.getFrameBlock(self.myId)
		
		self.processBatch(frames)

		print("Updated the graph")
		self.graph


	def processBatch(self,fr):
		print("NN in action")
