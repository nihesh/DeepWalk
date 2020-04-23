# Author : Nihesh Anderson 
# Date 	 : 21 March, 2020
# File 	 : train.py

NODE_PATH = "./data/nodes.csv"
EDGE_PATH = "./data/edges.csv"
DUMP = "./dump/"
WALK_LENGTH = 40
EPOCHS = 1000
BATCH_SIZE = 20
EMBED_SIZE = 128
SUBSAMPLE_SIZE = 1000			# 0 corresponds to the size of the dataset
LEARNING_RATE = 1e-3
WINDOW_SIZE = 5
SEED = 0
TEMPERATURE = 1

""" 
Best Parameters
walk length = 40
epochs = 100
batch_size = 20
embed_size = 32
subsample_size = 100
learning_rate = 1e-2
window_size = 10
seed = 0
temperature = 2
"""

import src.dataset as dataset
import torch
from tqdm import tqdm
import src.loss as loss
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import pickle
import os
import src.utils as utils
import random
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def ResetWorkspace():

	random.seed(SEED)
	np.random.seed(SEED)
	torch.manual_seed(SEED)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

if(__name__ == "__main__"):

	ResetWorkspace()

	dataset = dataset.Blog(NODE_PATH, EDGE_PATH, WALK_LENGTH)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 0)

	if(SUBSAMPLE_SIZE == 0):
		SUBSAMPLE_SIZE = len(dataset)

	num_samples = len(dataset)
	embedding = nn.Parameter(Variable(torch.randn([num_samples, EMBED_SIZE]).to(device), requires_grad = True))

	optimiser = optim.Adam([embedding], lr = LEARNING_RATE)

	for epoch in range(1, EPOCHS + 1):

		epoch_loss = 0
		num_samples = len(dataset)

		for walk in dataloader:

			walk = walk.to(device)
			batch_size = walk.shape[0]

			subsample = utils.sample(embedding, SUBSAMPLE_SIZE)
			norm_embedding = embedding / torch.norm(embedding, 2, dim = 1).unsqueeze(1)
			error = -loss.JointCooccurrenceLikelihood(walk, norm_embedding, subsample, WINDOW_SIZE, TEMPERATURE, device)

			optimiser.zero_grad()
			error.backward()
			optimiser.step()

			epoch_loss += -error.item() * batch_size


		print("[ Epoch {epoch} ] - Likelihood: {loss}".format(
				epoch = epoch, 
				loss = epoch_loss / num_samples
			))

		save = embedding.cpu().detach().numpy()

		file = open(os.path.join(DUMP, "embedding.pkl"), "wb")
		pickle.dump(save, file)
		file.close()