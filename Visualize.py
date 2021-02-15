import argparse
from Graph import *
from Person import *
import matplotlib.pyplot as plt
if __name__=="__main__":
	args=argparse.ArgumentParser()
	args.add_argument("--input","-i",type=str,dest="input")
	args.add_argument("--person","-p",type=int,dest="person")
	args=args.parse_args()
	print("---")
	graph=Graph()
	graph.loadFromFile(args.input)
	p = graph.getNode(args.person)
	p = p.params

	plt.scatter(p["X"],p["Y"])

	for a in range(min(len(p["X"]),len(p["Y"]))-1):
		plt.arrow(p["X"][a],p["Y"][a],\
			p["X"][a+1]-p["X"][a],p["Y"][a+1]-p["Y"][a])

	plt.show()

	plt.figure()

	k=p.keys()
	k=k-"X"
	k=k-"Y"
	print(k)

	plt.show()	
