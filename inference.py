# Author : Nihesh Anderson
# File   : inference.py
# Date 	 : 23 March, 2020

ROOT = "./dump/embedding.pkl"
LABELS = "./data/group-edges.csv"
SPLIT = 0.2							# Train : Test ratio

import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def ReadData(root):

	file = open(root, "rb")
	data = pickle.load(file)
	file.close()

	return data

def ReadLabels(root):

	file = open(root, "r")
	data = []

	num_samples = 0
	for line in file:
		line = line.split(",")
		num_samples = max(num_samples, int(line[0]))
		data.append(line)

	labels = [-1 for i in range(num_samples)]
	for line in data:
		labels[int(line[0]) - 1] = int(line[1]) - 1

	return np.asarray(labels)

if(__name__ == "__main__"):

	data = ReadData(ROOT)
	labels = ReadLabels(LABELS)

	num_samples = len(data)
	idx = [i for i in range(num_samples)]
	np.random.shuffle(idx)

	train = idx[:int(num_samples * SPLIT)]
	test = idx[int(num_samples * SPLIT):]

	train_label = labels[train]
	test_label = labels[test]

	train_data = data[train]
	test_data = data[test]

	model = SVC(C = 1.0, kernel = "rbf")
	model.fit(train_data, train_label)

	train_prediction = model.predict(train_data)
	test_prediction = model.predict(test_data)
	print("Train accuracy: {train_acc}".format(
			train_acc = accuracy_score(train_label, train_prediction)
		))
	print("Test accuracy: {test_acc}".format(
			test_acc = accuracy_score(test_label, test_prediction)
		))