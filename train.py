# Author : Nihesh Anderson 
# Date 	 : 21 March, 2020
# File 	 : train.py

ROOT = "./data/"
DUMP = "./dump/"
WALK_LENGTH = 10
EPOCHS = 3000
BATCH_SIZE = 20
EMBED_SIZE = 2
SUBSAMPLE_SIZE = 0
LEARNING_RATE = 1e-3
WINDOW_SIZE = 3
SEED = 0

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

	dataset = dataset.Blog(ROOT, WALK_LENGTH)
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
			error = -loss.JointCooccurrenceLikelihood(walk, embedding, subsample, WINDOW_SIZE, device)

			optimiser.zero_grad()
			error.backward()
			optimiser.step()

			epoch_loss += -error.item() * batch_size


		print("[ Epoch {epoch} ] - Likelihood: {loss}".format(
				epoch = epoch, 
				loss = epoch_loss / num_samples
			))

	embedding = embedding.cpu().detach().numpy()

	file = open(os.path.join(DUMP, "embedding.pkl"), "wb")
	pickle.dump(embedding, file)
	file.close()