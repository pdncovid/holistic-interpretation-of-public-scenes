import cv2 as cv

class InputHandler:
	def __init__(self):
		print("Created input handle")




	def setInputFile(self,fileName):
		self.inputSet=True
		self.videoIn=cv.VideoCapture(fileName)
		self.bufferZerothFrame=-1
		self.buffer=[]
		self.noConnectedComponents=0
		self.connectedCoponentBlockSizes=[]
		self.connectComponentNextFrame=[]

				
		# readOneFrameToBuffer(self)
		
	def readOneFrameToBuffer(self):
		ret,fr=self.videoIn.read()
		if ret:
			self.buffer.append(fr)
			self.bufferZerothFrame+=1
		else:
			print("ERROR!!!!")


	def connectComponent(self,framesPerBlock):
		#Returns the connected component ID
		thisNNID=self.noConnectedComponents
		self.noConnectedComponents+=1
		self.connectedCoponentBlockSizes.append(framesPerBlock)
		self.connectComponentNextFrame.append(0)

		print("Connected NN {} with block size {}".format(thisNNID,self.connectedCoponentBlockSizes[thisNNID]))
		return thisNNID
		#Connected component ID

	def cleanBuffer(self):
		print("Dummy buffer cleaner")
		

	def getFrameBlock(self,requesterID):
		requestZerothFrame=self.connectComponentNextFrame[requesterID]
		requestLastFrame=requestZerothFrame+self.connectedCoponentBlockSizes[requesterID]
		
		while self.bufferZerothFrame+len(self.buffer) < requestLastFrame:
			self.readOneFrameToBuffer()

		self.cleanBuffer()

		toReturn= self.buffer[requestZerothFrame-self.bufferZerothFrame:requestZerothFrame-self.bufferZerothFrame]
		print("Returning {} frames".format(len(toReturn)))
		return toReturn