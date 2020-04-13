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
Tensorf = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


import args
import pickle
import sys
import matplotlib.pyplot as plt
grad_counter = 0
import os
def nll(prob):
    return torch.mean(-1 * torch.log(prob))
            
def write_walk(random_walk, tree, f, data_counter):
    for idx in range(len(random_walk)):
        for j in range(max(0, idx-args.window_length), min(len(random_walk), idx + args.window_length + 1)):
            if(idx == j):
                continue
            input_node_idx = random_walk[j]
            context_idx = random_walk[idx]
            index_path_tree, binary_multipliers = tree.get_path(input_node_idx)
            f[str(data_counter)] = np.array([context_idx] + index_path_tree + binary_multipliers + [input_node_idx])
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
    g = Graph("../data/nodes.csv", "../data/edges.csv", subset_size=args.subset_size)
    tree = SoftmaxTree(embed_size = args.embed_size)
    if(args.tree_constructor == "huffman"):
        degreelist = g.get_degrees()
        degreelist = sorted(degreelist, key = lambda x : x[1])
        tree.create_huffman(degreelist)
    elif(args.tree_constructor == "complete"):
        vertices = g.get_vertices()
        tree.create_complete_tree(vertices)
    tree = tree.to(device)
    embeddings = EmbeddingMatrix(num_nodes = g.num_nodes, embed_size = args.embed_size).to(device)
    embeddings2 = np.copy(list(embeddings.parameters())[0].data.cpu().numpy())
    optimE = optim.Adam(list(embeddings.parameters()) + list(tree.parameters()), lr = args.lr)
    if(not args.use_old_copy):
        g.save_graph("./graph.pkl")
        deepwalk(tree,g)
    print(g.num_nodes)
    
    dataset = WalkPairData(args.hdf5file)
    train_loader = DataLoader(dataset, batch_size = args.batch_size, num_workers = 1, shuffle = True)
    totalsize = len(train_loader)
    for e in range(1):
        avg_loss = 0
        count = 0
        for idx,batch in (enumerate(train_loader)):
            batch = Variable(batch.type(Tensor), requires_grad = False)
            context_idx = batch[:,0]
            length = batch.shape[1] - 1
            length //= 2
            tree_path_idx = batch[:,1:1+length]
            binary_multipliers = Variable(batch[:,1+length:1+2*length].type(Tensorf), requires_grad = False)
            optimE.zero_grad()
            context_vector = embeddings(context_idx)
            prob = tree(context_vector, tree_path_idx,binary_multipliers)
            loss = nll(prob)
            print(e, idx, totalsize,loss,prob)
            loss.backward()
            avg_loss += loss.item()
            count += 1
            #for name,p in embeddings.named_parameters():
            #   print(p.grad)
            optimE.step()
        print(e, avg_loss/count)
    embeddings = np.copy(list(embeddings.parameters())[0].data.cpu().numpy())
    f = open("embeddings.pkl", "wb")
    pickle.dump((embeddings,embeddings2), f)
    f.close()


