import argparse

from Graph import *
from Person import *
from InputHandler import *

from NNHandler_openpose import *
from NNHandler_yolo import *


if __name__=="__main__":
	args=argparse.ArgumentParser()
	args.add_argument("--input","-i",type=str,dest="input")
	args=args.parse_args()

	graph= Graph()


	cctv=InputHandler()
	cctv.setInputFile(args.input)

	nn=[NNHandler_yolo(), NNHandler_openpose()]

	for n in nn:
		n.setInputBlockSize(32)
		n.connectToInput(cctv)
		n.connectToGraph(graph)


	
	for n in nn:
		cctv.connectComponent(n)

	#>>> Naive scheduling>>>>>>

	for ITER in range(2):
		for n in nn:
			n.runForBatch()

	graph.saveToFile()





	