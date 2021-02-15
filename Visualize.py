import argparse
from Graph import *
from Person import *
import matplotlib.pyplot as plt
import numpy as np
if __name__=="__main__":
	args=argparse.ArgumentParser()
	args.add_argument("--input","-i",type=str,dest="input")
	args.add_argument("--person","-p",type=int,dest="person")
	args.add_argument("--output","-o",type=str,dest="output")
	args=args.parse_args()
	print("---")
	graph=Graph()
	graph.loadFromFile(args.input)
	p = graph.getNode(args.person)
	p = p.params


	fig, axs = plt.subplots(2)
	fig.suptitle('Information of person : {}'.format(args.person))
	# axs[0].plot(x, y)
	# axs[1].plot(x, -y)

	axs[0].scatter(p["X"][0],p["Y"][0],color='r')
	axs[0].scatter(p["X"][1:-1],p["Y"][1:-1],color='g')
	axs[0].scatter(p["X"][-1],p["Y"][-1],color='b')
	axs[0].legend(["Start","Path","End"])

	for a in range(min(len(p["X"]),len(p["Y"]))-1):
		axs[0].arrow(p["X"][a],p["Y"][a],\
			p["X"][a+1]-p["X"][a],p["Y"][a+1]-p["Y"][a])

	# plt.show()

	# plt.figure()

	keys=list(p.keys())
	keys.remove("X")
	keys.remove("Y")
	print(keys)

	toPlot=[]
	for k in keys:
		ar=np.array(p[k],dtype=np.float)
		ar=ar-np.min(ar)
		ar=ar/np.max(ar)
		toPlot.append(ar)
		axs[1].plot(np.arange(len(ar)),ar)

	axs[1].legend(keys)
	print(args.output)
	plt.show()	
