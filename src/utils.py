# Author : Nihesh Anderson
# Date   : 21 March, 2020
# File 	 : utils.py

import random

def random_walk(node, graph, size):

	"""
	Returns a random walk of mentioned size uniformly at random from the given graph
	Recursive implementation
	"""

	if(size == 0): 
		
		return []

	sz = len(graph[node])
	nxt = random.randint(0, sz - 1)

	result = random_walk(graph[node][nxt], graph, size - 1)
	result.append(node)

	return result


if(__name__ == "__main__"):

	pass