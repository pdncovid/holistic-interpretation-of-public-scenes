import argparse
from Graph import *
from Node_Person import Person
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la



if __name__=="__main__":
	args=argparse.ArgumentParser()
	args.add_argument("--input","-i",type=str,dest="input")
	args.add_argument("--outputPrefix","-op",type=str,dest="outputPrefix")
	args.add_argument("--onlyDetectedTime",type=bool,dest="onlyDetectedTime"\
		,default=False)
	args.add_argument("--interpolateUndetected",type=bool,\
		dest="interpolateUndetected",default=False)

	args.add_argument("--proximityThreshold",type=float,dest="proximityThreshold",default=1.00)
	args=args.parse_args()



	graph=Graph()
	graph.init_from_json(args.input)
	graph.calculate_standing_locations()
	graph.interpolate_undetected_timestamps()


	N=len(graph.nodes)
	T=graph.time_series_length


	handshakeMatrix=np.zeros((N,N,T),dtype=np.bool)
	proximitryMatrix=np.zeros((N,N,T),dtype=np.float)
	standingLocationXY=np.zeros((N,T,2),dtype=np.float)
	interPersonalDistance=np.zeros((N,N,T),dtype=np.float)

	for idx_p1 in range(len(graph.nodes)):#First person under consideration
		p1=graph.nodes[idx_p1]
		for t in range(len(p1.params["handshake"])):
			h_t=p1.params["handshake"][t]#handshake at time t
			# print("DEBUG: ",h_t["person"])
			if h_t["person"]!=None:
				p2=h_t["person"]#Second person under consideration
				# print("DEBUG --: ",idx_p1,p2,t)
				# print("DEBUG --: ",type(idx_p1),type(p2),type(t))
				# print("DEBUG --: ",handshakeMatrix[idx_p1,p2,t])
				handshakeMatrix[idx_p1,p2,t]=True


	for idx_p1 in range(len(graph.nodes)):
		p1=graph.nodes[idx_p1]
		xt=p1.params["X"]
		yt=p1.params["Y"]
		for t in range(len(xt)):
			standingLocationXY[idx_p1,t,0]=xt[t]
			standingLocationXY[idx_p1,t,1]=yt[t]

	for idx_p1 in range(N):
		for idx_p2 in range(N):
			for t in range(T):
				interPersonalDistance[idx_p1,idx_p2,t]=\
				la.norm(standingLocationXY[idx_p1,t]-standingLocationXY[idx_p2,t],ord=2)



	handshakeMatrix_atLeastOnce=np.zeros((N,N),dtype=np.bool)
	for p1 in range(N):
		for p2 in range(N):
			handshakeMatrix_atLeastOnce[p1,p2]=np.any(handshakeMatrix[p1,p2,:])

			if p1==p2:
				handshakeMatrix_atLeastOnce[p1,p2]=True

	handshakeMatrix_atLeastOnce=handshakeMatrix_atLeastOnce.astype(np.float)
	print("------------\n\nhandshakeMatrix_atLeastOnce")
	print(handshakeMatrix_atLeastOnce)

	plt.imshow(handshakeMatrix_atLeastOnce, interpolation='none')
	plt.xlabel("person 1 ID")
	plt.ylabel("person 2 ID")
	plt.title("Shook hands at least once")
	plt.show()

	clusterMatrix=handshakeMatrix_atLeastOnce
	for a in range(N):
		clusterMatrix=np.dot(clusterMatrix,handshakeMatrix_atLeastOnce)
	print("------------\n\nclusterMatrix")
	print(clusterMatrix)

	plt.figure()
	plt.imshow(handshakeMatrix_atLeastOnce, interpolation='none')
	plt.xlabel("person 1 ID")
	plt.ylabel("person 2 ID")
	plt.title("Proximity")
	plt.show()

	# plt.legend(["aa","bb"])








	