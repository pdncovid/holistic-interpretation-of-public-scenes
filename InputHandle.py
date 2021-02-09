import cv2 as cv

class InputHandle:
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

		readOneFrameToBuffer()
		
	def readOneFrameToBuffer(self):
		ret,fr=self.videoIn.read()
		if ret:
			self.buffer.append(fr)
			self.bufferZerothFrame+=1
		else:
			print("ERROR!!!!")

	def connectComponent(self,framesPerBlock):
		#Returns the connected component ID
		self.noConnectedComponents+=1
		self.connectedCoponentBlockSizes.append(framesPerBlock)
		self.connectComponentNextFrame.append(0)
		return self.noConnectedComponents-1#Connected component ID

	def cleanBuffer():
		

	def getFrameBlock(self,requesterID):
		requestZerothFrame=self.connectComponentNextFrame[requesterID]
		requestLastFrame=requestZerothFrame+self.connectedCoponentBlockSizes[requesterID]
		
		while self.bufferZerothFrame+len(self.buffer) < requestLastFrame:
			readOneFrameToBuffer()

		cleanBuffer()