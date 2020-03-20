# Author : Nihesh Anderson 
# Date 	 : 21 March, 2020
# File 	 : dataset.py

from torch.utils.data import Dataset
import os
import numpy as np
import src.utils as utils

class Blog(Dataset):

	def __init__(self, root, subset_size):

		self.root = root
		self.subset_size = subset_size

		node_file = open(os.path.join(root, "nodes.csv"), "r")
		nodes = []
		for line in node_file:
			nodes.append(int(line))
		node_file.close()

		self.num_nodes = np.max(nodes)
		self.graph = [[] for i in range(self.num_nodes + 1)]

		# Build Graph
		edge_file = open(os.path.join(root, "edges.csv"), "r")
		for line in edge_file:
			u, v = list(map(int, line.split(",")))
			self.graph[u].append(v)
			self.graph[v].append(u)
		edge_file.close()

	def __len__(self):

		return self.num_nodes

	def __getitem__(self, node):

		"""
		node is the idx starting from 0 to num_nodes - 1. The graph structure is indexed from 1 onwards
		"""

		print(node, self.num_nodes)

		return utils.random_walk(node + 1, self.graph, self.subset_size)

if(__name__ == "__main__"):

	pass