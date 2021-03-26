import argparse

from Graph import *
from Node_Person import Person
from InputHandler import *

from NNHandler_handshake import *
from NNHandler_yolo import *


if __name__=="__main__":
	args=argparse.ArgumentParser()
	args.add_argument("--input","-i",type=str,dest="input")
	args.add_argument("--saveGraph","-sg",type=str,dest="saveGraph",default=None)
	args.add_argument("--nnout_yolo",type=str,dest="nnout_yolo")
	args.add_argument("--nnout_handshake",type=str,dest="nnout_handshake")
	args.add_argument("--timeSeriesLength",type=int,dest="timeSeriesLength",default=1000)
	args=args.parse_args()


	graph= Graph(save_name=args.saveGraph,\
		time_series_length=args.timeSeriesLength)


	cctv=InputHandler()
	if args.input!=None:
		cctv.setInputFile(args.input)

	nn=[NNHandler_yolo(textFileName=args.nnout_yolo),\
	 NNHandler_handshake(args.nnout_handshake)]

	for n in nn:
		n.setInputBlockSize(32)
		# n.connectToInput(cctv)
		n.connectToGraph(graph)

	
	# for n in nn:
	# 	cctv.connectComponent(n)

	#>>> Naive scheduling>>>>>>


	for ITER in range(1):
		for n in nn:
			n.runForBatch()

	if args.saveGraph!=None:
		graph.saveToFile()





	