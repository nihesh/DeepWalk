# Author : Nihesh Anderson
# Date 	 : April 13, 2020
# File 	 : visualise.py

import numpy as np
import pickle
from matplotlib import pyplot as plt
from collections import deque
from sklearn.manifold import TSNE

def Visualisekarate(file_name, save_path):

	file = open(file_name, "rb")

	embedding = pickle.load(file)
	colors = ['red','red','green','red','blue','blue','blue','red','purple','green','blue','red','red','red','purple','purple','blue','red','purple','red','purple','red','purple','purple','green','green','purple','green','green','purple','purple','green','purple','purple']

	plt.clf()
	plt.scatter(embedding[:, 0], embedding[:, 1], c = colors)
	plt.savefig(save_path)

def VisualiseSeparation(source_node, d_close, d_far, file_name, save_path):

	# Load embeddings
	file = open(file_name, "rb")
	embedding = pickle.load(file)
	embedding = np.asarray(embedding)

	# Load graph
	node_path = "./data/nodes.csv"
	edge_path = "./data/edges.csv"

	# Retrieve node list
	node_file = open(node_path, "r")
	nodes = []
	for line in node_file:
		nodes.append(int(line))
	node_file.close()

	num_nodes = np.max(nodes)
	graph = [[] for i in range(num_nodes + 1)]

	# Build Graph
	edge_file = open(edge_path, "r")
	for line in edge_file:
		u, v = list(map(int, line.split(",")))
		graph[u].append(v)
		graph[v].append(u)
	edge_file.close()
	
	# BFS to find distance of nodes from source
	q = deque()
	q.append((source_node, 0))
	dist = [1000000000 for i in range(num_nodes + 1)]

	while(len(q)):
	
		now = q.popleft()
		if(dist[now[0]] != 1000000000):
			continue
		dist[now[0]] = now[1]

		for child in graph[now[0]]:
			q.append((child, now[1] + 1))

	# Segregate nodes as close and far
	close_nodes = []
	far_nodes = []
	for i in range(1, num_nodes + 1):
		if(dist[i] <= d_close):
			close_nodes.append(i)
		if(dist[i] >= d_far):
			far_nodes.append(i)
	labels = []

	close_nodes = np.asarray(close_nodes)
	far_nodes = np.asarray(far_nodes)

	# Generate data and corresponding label for plotting
	data = np.concatenate([embedding[close_nodes - 1], embedding[far_nodes - 1]], axis = 0)
	for i in range(close_nodes.shape[0]): 
		labels.append(0)
	for i in range(far_nodes.shape[0]):
		labels.append(1)

	# Plot TSNE
	data = TSNE().fit_transform(data)
	plt.clf()
	plt.scatter(data[:, 0], data[:, 1], c = labels)
	plt.savefig(save_path)

if(__name__ == "__main__"):

	# VisualiseKarate("./dump/embedding.pkl", "./results/karate.png")
	VisualiseSeparation(1, 1, 3, "./dump/embedding.pkl", "./results/close_far_visualisation.png")