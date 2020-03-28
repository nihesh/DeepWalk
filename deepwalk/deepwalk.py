from graph import Graph,WalkPairData
from parameters import SoftmaxTree, EmbeddingMatrix
from tqdm import tqdm
import torch
import h5py
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable
import random
import torch.optim as optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Tensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
import args
import pickle
import sys
import matplotlib.pyplot as plt
grad_counter = 0
import os
def nll(prob):
    return -1 * torch.log(prob)
            
def write_walk(random_walk, tree, f, data_counter):
    for idx in range(len(random_walk)):
        for j in range(max(0, idx-args.window_length), min(len(random_walk), idx + args.window_length + 1)):
            input_node_idx = random_walk[j]
            context_idx = random_walk[idx]
            index_path_tree = tree.get_path(input_node_idx)
            x = np.array([context_idx] + index_path_tree)
            f[str(data_counter)] = x
            data_counter += 1 
    return data_counter
def deepwalk(tree,graph):
    data_counter = 0
    f = h5py.File(args.hdf5file, "w")
    nodes_index = [i for i in range(graph.num_nodes)]
    for gamma in (range(1, args.walks_per_vertex+1)):
        random.shuffle(nodes_index)
        for counter,vi in (enumerate(nodes_index)):
            print("iter {} vertex {}".format(gamma, counter))
            random_walk = graph.random_walk(counter, args.walk_length)
            data_counter = write_walk(random_walk, tree, f, data_counter)
    f.close()
if __name__ == "__main__":
    g = Graph("../data/karate_nodes.txt", "../data/karate_edges.txt")
    degreelist = g.get_degrees()
    degreelist = sorted(degreelist, key = lambda x : x[1])
    tree = SoftmaxTree(embed_size = args.embed_size)
    tree.create_huffman(degreelist)
    tree = tree.to(device)
    
    embeddings = EmbeddingMatrix(max_nodes = args.max_nodes, embed_size = args.embed_size).to(device)
    embeddings2 = np.copy(list(embeddings.parameters())[0].data.cpu().numpy())
    optimE = optim.Adam(embeddings.parameters(), lr = args.lr)
    optimT = optim.Adam(tree.parameters(), lr = args.lr)
    deepwalk(tree,g)
    dataset = WalkPairData(args.hdf5file)
    train_loader = DataLoader(dataset, batch_size = 1, num_workers = 1, shuffle = True)
    totalsize = len(train_loader)
    for e in range(1):
        for idx,batch in (enumerate(train_loader)):
            batch = Variable(batch.type(Tensor), requires_grad = False)
            context_idx = batch[:,0]
            tree_path_idx = batch[:,1:].squeeze(0)
            optimE.zero_grad()
            optimT.zero_grad()
            context_vector = embeddings(context_idx)
            prob = tree(context_vector, tree_path_idx)
            loss = nll(prob)
            print(idx, totalsize,loss,prob)
            loss.backward()
            #for name,p in tree.named_parameters():
            #   print(p.grad)
            optimT.step()
            optimE.step()
    embeddings = np.copy(list(embeddings.parameters())[0].data.cpu().numpy())
    f = open("embeddings.pkl", "wb")
    pickle.dump((embeddings,embeddings2), f)
    f.close()


