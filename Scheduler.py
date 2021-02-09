import argparse

from Graph import *
from Person import *
from InputHanler import *

from NNHandler_openpose import *
from NNHandler_yolo import *

from arg

if __name__=="__main__":
	args=argparse.ArgumentParser()
	args.add_argument("--input","-i",type=str,dest="input")
	args=args.parse_args()

	graph= Graph()


	nn=[NNHandler_yolo(), NNHandler_openpose()]

	cctv=InputHandler()
	cctv.setInputFile(args.input)
	
	for n in nn:
		cctv.connectComponent(n)

	#>>> Naive scheduling>>>>>>

	for ITER in range(1):
		for n in nn:
			n.run()





	