# Author : Nihesh Anderson
# Date 	 : 22 March, 2020
# File 	 : loss.py

import torch
import src.utils as utils

def JointCooccurrenceLikelihood(random_walk, embedding):

	"""
	Computes the joint cooccurence loss averaged over the batches and returns it
	"""

	batch_size = random_walk.shape[0]

	random_walk = random_walk - 1
	selected_embeddings = embedding[random_walk]
	n = selected_embeddings.shape[1]

	first_few = selected_embeddings[:, : n - 1, :]
	last_few = selected_embeddings[:, 1:, :]

	similarity = first_few * last_few
	similarity = similarity / torch.norm(first_few, 2, dim = 2).unsqueeze(2)
	similarity = similarity / torch.norm(last_few, 2, dim = 2).unsqueeze(2)
	similarity = similarity.sum(dim = 2)
	similarity = (similarity + 1) / 2

	# Evade multiplication by adding log probabilities
	joint = torch.log(similarity)
	joint = joint.sum(dim = 1)

	joint = joint.sum(dim = 0)
	utils.nan_check(joint)

	return joint / batch_size

if(__name__ == "__main__"):

	pass