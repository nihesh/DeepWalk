# Author : Nihesh Anderson
# Date 	 : April 13, 2020
# File 	 : visualise.py

import pickle
from matplotlib import pyplot as plt

if(__name__ == "__main__"):

	file = open("./dump/embedding.pkl", "rb")

	embedding = pickle.load(file)
	colors = ['red','red','green','red','blue','blue','blue','red','purple','green','blue','red','red','red','purple','purple','blue','red','purple','red','purple','red','purple','purple','green','green','purple','green','green','purple','purple','green','purple','purple']

	plt.clf()
	plt.scatter(embedding[:, 0], embedding[:, 1], c = colors)
	plt.savefig("./results/karate.png")