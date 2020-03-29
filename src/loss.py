# Author : Nihesh Anderson
# Date 	 : 22 March, 2020
# File 	 : loss.py

import torch
import src.utils as utils

def JointCooccurrenceLikelihood(random_walk, embedding, subsample):

	"""
	Computes the joint cooccurence loss averaged over the batches and returns it
	"""

	batch_size = random_walk.shape[0]

	subsample = subsample.unsqueeze(0).repeat(batch_size, 1, 1)

	random_walk = random_walk - 1
	selected_embeddings = embedding[random_walk]
	n = selected_embeddings.shape[1]

	first_few = selected_embeddings[:, : n - 1, :]
	last_few = selected_embeddings[:, 1:, :]

	pairwise_dot = (first_few.unsqueeze(2) * subsample.unsqueeze(1)).sum(dim = 3)
	pairwise_dot = pairwise_dot / torch.norm(first_few, 2, dim = 2).unsqueeze(2)
	pairwise_dot = pairwise_dot / torch.norm(subsample, 2, dim = 2).unsqueeze(1)
	pairwise_dot = (pairwise_dot + 1) / 2
	normalisation_factor = pairwise_dot.sum(dim = 2)


	similarity = first_few * last_few
	similarity = similarity / torch.norm(first_few, 2, dim = 2).unsqueeze(2)
	similarity = similarity / torch.norm(last_few, 2, dim = 2).unsqueeze(2)
	similarity = similarity.sum(dim = 2)
	similarity = (similarity + 1) / 2

	# Normalise similarity using normalisation factor
	similarity = similarity / (similarity + normalisation_factor)

	# Evade multiplication by adding log probabilities
	joint = torch.log(similarity)
	joint = joint.sum(dim = 1)

	joint = joint.sum(dim = 0)
	utils.nan_check(joint)

	return joint / batch_size

if(__name__ == "__main__"):

	pass