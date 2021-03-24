import argparse
from Graph import *
from Node_Person import Person
import matplotlib.pyplot as plt
import numpy as np

if __name__=="__main__":
	args=argparse.ArgumentParser()
	args.add_argument("--input","-i",type=str,dest="input")
	args.add_argument("--person","-p",type=str,dest="person")
	args.add_argument("--output","-o",type=str,dest="output")
	args=args.parse_args()
	print("---")
	graph=Graph()
	graph.init_from_json(args.input)
	graph.calculate_standing_locations()
	

	if "," not in args.person:
		args.person=int(args.person)
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
		for k in ["X","Y","xMin","xMax","yMin","yMax","handshake"]:
			keys.remove(k)
		print(keys)

		toPlot=[]
		for k in keys:
			ar=np.array(p[k],dtype=np.float)
			ar=ar-np.min(ar)
			ar=ar/np.max(ar)
			toPlot.append(ar)
			axs[1].plot(np.arange(len(ar)),ar)

		axs[1].legend(keys)
		
	else:
		pp=list(map(int,args.person.strip().split(",")))
		maxLen=0
		for p in pp:
			p=graph.getNode(p).params
			maxLen=max(maxLen,max(len(p["X"]),len(p["Y"])))
		locX=np.zeros((len(pp),maxLen),dtype=np.float)
		locY=np.zeros((len(pp),maxLen),dtype=np.float)
		for p in range(len(pp)):
			person=graph.getNode(pp[p]).params
			locX[p,:len(person["X"])]=np.array(person["X"],dtype=np.float)
			locY[p,:len(person["Y"])]=np.array(person["Y"],dtype=np.float)

		cogX=np.mean(locX,axis=0)
		cogY=np.mean(locY,axis=0)


		print(cogX,cogY)
		fig, axs = plt.subplots(2)

		legLine=[]
		legLine.append(axs[0].scatter(cogX[0],cogY[0],color='r'))
		legLine.append(axs[0].scatter(cogX[1:-1],cogY[1:-1],color='g'))
		legLine.append(axs[0].scatter(cogX[-1],cogY[-1],color='b'))
		
		for a in range(cogX.shape[0]-1):
			axs[0].arrow(cogX[a],cogY[a],\
				cogX[a+1]-cogX[a],cogY[a+1]-cogY[a])

		
		legWord=["Start","Path","End"]
		for p in range(len(pp)):
			person=graph.getNode(pp[p]).params
			x=person["X"]
			y=person["Y"]
			minLen=min(len(x),len(y))
			x=x[:minLen]
			y=y[:minLen]
			legLine.append(axs[0].scatter(x,y,marker="."))
			for a in range(minLen):
				axs[0].arrow(cogX[a],cogY[a],x[a]-cogX[a],y[a]-cogY[a],
					linestyle="dotted")
			legWord.append("P{}".format(p))
		axs[0].legend(legLine,legWord)

		
		distX=np.array(locX)
		distY=np.array(locY)
		dist=distX.fill(0.0)
		for n in range(len(graph.nodes)):
			distX[n,:]=distX[n,:]-cogX
			distY[n,:]=distY[n,:]-cogY
		# print(distX.shape)
		dist=np.sqrt(np.square(distX)+np.square(distY))

		legLine=[]
		legWord=[]
		
		axs[1].plot(np.mean(dist,axis=0))
		legWord.append("Group dist from COG")

		for d in range(dist.shape[0]):
			axs[1].plot(dist[d],":")
			legWord.append("P {}".format(d))
		axs[1].legend(legWord)

		print("Dist",dist)


	if args.output==None:
		plt.show()	
	else:
		plt.savefig(args.output)


		# print(pp)
