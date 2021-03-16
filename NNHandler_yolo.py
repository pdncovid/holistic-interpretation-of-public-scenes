from NNHandler import *
import numpy as np
class NNHandler_yolo(NNHandler):
	def __init__(self,textFileName="yoloOut.txt",N=256):
		print("Creating an Yolo handler")
		self.fileName=textFileName
		self.file=open(self.fileName,mode='r')
		self.inputBlockSize=N
		self.allLines=self.file.readlines()
		# print(self.allLines)
		self.file.close()
	
	def extractValForKey(self,st,startSt,endSt):
		a=st.index(startSt)+len(startSt)
		b=st.index(endSt)
		return st[a:b].strip()

	




	def runForBatch(self):
		print("Running....")
		
		frames=[]
		for l in self.allLines[1:]:
			if l.split(" ")[0]=="Frame":
				frames.append([])
			elif l.split(" ")[0]=="FPS:":
				pass
			elif l.split(" ")[0]=="Tracker":
				# print()
				try:
					a="Tracker ID:"
					b="Class:"
					c="BBox Coords (xmin, ymin, xmax, ymax):"
					d="\n"
					frames[-1].append(dict())
					# print(frames[-1])
					frames[-1][-1]["id"]=int(self.extractValForKey(l,a,b)[:-1])
					frames[-1][-1]["class"]=self.extractValForKey(l,b,c)
					frames[-1][-1]["bb"]=list(map(int,self.extractValForKey(l,c,d)[1:-1].split(",")))
				except:
					break


		# print(len(frames),len(frames[300]),frames[5])
		ids=[]
		for f in frames:
			for o in f:
				ids.append(o["id"])
		ids=set(ids)


		print("UniqueIDs ",sorted(ids))



		# self.myInput()


		# print(self.allLines)

		print("Updated the graph")



	if __name__=="__main__":
		print("Testing started >>>>>>")
		a=extractValForKey("Tracker ID: 15, Class: person,  BBox Coords (xmin, ymin, xmax, ymax): (1154, 0, 1194, 75)\n","Tracker ID:",", Class")
		print(a)

		a=extractValForKey("Tracker ID: 15, Class: person,  BBox Coords (xmin, ymin, xmax, ymax): (1154, 0, 1194, 75)\n","BBox Coords (xmin, ymin, xmax, ymax):","\n")
		print(a)

		a=extractValForKey("Tracker ID: 15, Class: person,  BBox Coords (xmin, ymin, xmax, ymax): (1154, 0, 1194, 75)\n","Class:",",  BBox")
		print(a)


		print(">>>>>> Testing ended")