# Author : Nihesh Anderson
# Date 	 : 22 March, 2020
# File 	 : loss.py

import torch
import src.utils as utils
from src.constants import EPS

def NormalisationFactor(fixed_pts, subsample):

	"""
	Returns the normalisation denominator for probability computation
	"""

	pairwise_dot = (fixed_pts.unsqueeze(2) * subsample.unsqueeze(1)).sum(dim = 3)
	pairwise_dot = pairwise_dot / torch.norm(fixed_pts, 2, dim = 2).unsqueeze(2)
	pairwise_dot = pairwise_dot / torch.norm(subsample, 2, dim = 2).unsqueeze(1)
	pairwise_dot = (pairwise_dot + 1) / 2
	normalisation_factor = pairwise_dot.sum(dim = 2)

	return normalisation_factor

def ShiftAndMult(fixed_pts, shift, normalisation_factor):

	num_samples = fixed_pts.shape[1]

	shifted = fixed_pts[:, max(shift, 0): min(num_samples - 1, num_samples + shift - 1)]
	base = fixed_pts[:, max(-shift, 0): min(num_samples - 1, num_samples - shift - 1)]

	similarity = shifted * base
	similarity = similarity.sum(dim = 2)
	similarity = similarity / torch.norm(shifted, 2, dim = 2)
	similarity = similarity / torch.norm(base, 2, dim = 2)
	similarity = (similarity + 1) / 2

	trimmed_norm = normalisation_factor[:, max(-shift, 0): min(num_samples - 1, num_samples - shift - 1)]
	similarity = similarity / (similarity + trimmed_norm)

	return similarity

def JointCooccurrenceLikelihood(random_walk, embedding, subsample, window_size, device):

	"""
	Computes the joint cooccurence loss averaged over the batches and returns it
	"""

	batch_size = random_walk.shape[0]

	subsample = subsample.unsqueeze(0).repeat(batch_size, 1, 1)

	random_walk = random_walk - 1
	selected_embeddings = embedding[random_walk]
	n = selected_embeddings.shape[1]

	fixed_pts = selected_embeddings

	normalisation_factor = NormalisationFactor(fixed_pts, subsample)

	loss = 0

	for i in range(-(window_size - 1), window_size + 1):

		if(i == 0):
			continue

		similarity = ShiftAndMult(fixed_pts, i, normalisation_factor)

		joint = torch.log(torch.max(similarity, torch.tensor([EPS]).float().to(device))).float().to(device)
		joint = joint.sum(dim = 1)
		joint = joint.sum(dim = 0)

		loss = loss + joint / batch_size

	utils.nan_check(loss)

	return loss

if(__name__ == "__main__"):

	pass