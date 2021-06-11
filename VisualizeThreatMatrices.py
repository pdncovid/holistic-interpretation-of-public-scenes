from Graph import *
import argparse
import matplotlib.pyplot as plt




if __name__=="__main__":
	args=argparse.ArgumentParser()
	args.add_argument("--outputPrefix","-o",type=str,dest="outputPrefix",default="./visualizations/threatmatrices/")
	args=args.parse_args()


	#This part of the code is directly taken from Graph.py:main
	#>> Done edit this without editing Graph.py >>>>>>>>>>>>>>
	g = Graph()
	# g.init_from_json('./data/vid-01-graph.json')		# Start from yolo
	g.init_from_json('./data/vid-01-graph_handshake.json')  # Start from handshake
	print("Created graph with nodes = %d for frames = %d. Param example:" % (g.n_nodes, g.time_series_length))
	g.fullyAnalyzeGraph()
	#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

	print("pairD",g.pairD.shape)
	print("pairI",g.pairI.shape)
	print("pairM",g.pairM.shape)
	print("pairG",g.pairG.shape)
	print("pairT",g.pairT.shape)
	print("frameThreatLevel",g.frameThreatLevel.shape)

	print("pairD",g.pairD)
	print("pairI",g.pairI)
	print("pairM",g.pairM)
	print("pairG",g.pairG)
	print("pairT",g.pairT)




	for t in range(100):#range(g.time_series_length):
		'''
			There is a big memory leak in this approach. Do you know how to fix?
		'''

		# plt.figure()
		plt.matshow(g.pairD[t,:,:])
		plt.colorbar()
		plt.savefig("{}d-{:04d}".format(args.outputPrefix,t))
		plt.close()

		# plt.figure()
		plt.matshow(g.pairI[t,:,:],vmin=0, vmax=1)
		plt.colorbar()
		plt.savefig("{}i-{:04d}".format(args.outputPrefix,t))		
		plt.close()

		# plt.figure()
		plt.matshow(g.pairM[t,:,:],vmin=0, vmax=1)
		plt.colorbar()
		plt.savefig("{}m-{:04d}".format(args.outputPrefix,t))		
		plt.close()

		# plt.figure()
		plt.matshow(g.pairG[t,:,:],vmin=0, vmax=1)
		plt.colorbar()
		plt.savefig("{}g-{:04d}".format(args.outputPrefix,t))		
		plt.close()

		# plt.figure()
		plt.matshow(g.pairT[t,:,:])
		plt.colorbar()
		plt.savefig("{}t-{:04d}".format(args.outputPrefix,t))		
		plt.close()