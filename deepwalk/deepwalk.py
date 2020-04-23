from graph import Graph
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
#dictionary = {}
def write_walk(random_walk, tree):
    global dictionary
    ret = None
    for idx in range(len(random_walk)):
        for j in range(max(0, idx-args.window_length), min(len(random_walk), idx + args.window_length + 1)):
            if(idx == j):
                continue
            input_node_idx = random_walk[j]
            context_idx = random_walk[idx]
            #index_path_tree, binary_multipliers = #tree.get_path(input_node_idx)
            #f[str(data_counter)] = 
            #dictionary[data_counter] = np.array([context_idx] + [input_node_idx]) 
            temp = np.array([context_idx] + [input_node_idx])
            if(ret is None):
                ret = temp
            else:
                ret = np.vstack([ret, temp])
    #print(ret.shape)
    return ret

def deepwalk(tree,graph):
    global dictionary
    data_counter = 0
    nodes_index = [i for i in range(graph.num_nodes)]
    for gamma in (range(1, args.walks_per_vertex+1)):
        random.shuffle(nodes_index)
        for counter,vi in (enumerate(nodes_index)):
            print("iter {} vertex {}".format(gamma, counter))
            random_walk = graph.random_walk(counter, args.walk_length)
            data_counter = write_walk(random_walk, tree, data_counter)

from torch.utils.data import Dataset
class WalkPairData(Dataset):
    def __init__(self, num_vertices, gamma, graph, tree):
        super(WalkPairData, self).__init__()
        self.num_vertices = num_vertices
        self.gamma = gamma
        self.graph = graph
        self.tree = tree
    def __len__(self):
        return self.num_vertices * self.gamma 
    def __getitem__(self, idx):
        vertex_id = idx % self.num_vertices
        random_walk = self.graph.random_walk(vertex_id, args.walk_length)
        ret = write_walk(random_walk, self.tree)
        return ret
    
if __name__ == "__main__":
    g = Graph("../data/nodes.csv", "../data/edges.csv", subset_size = args.subset_size)
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
    
    embeddings2 = np.copy(list(embeddings.parameters())[0].data.cpu().numpy()) #remove occurences later
    optimE = optim.Adam(list(embeddings.parameters()) + list(tree.parameters()), lr = args.lr)
    
    g.save_graph("./graph.pkl")
    #deepwalk(tree,g)
    dataset = WalkPairData(g.num_nodes, args.walks_per_vertex, g, tree)
    train_loader = DataLoader(dataset, batch_size = args.batch_size, num_workers = 1, shuffle = True)
    totalsize = len(train_loader)
    for e in range(1):
        avg_loss = 0
        count = 0
        for idx,batch in (enumerate(train_loader)):
            batch = Variable(batch.type(Tensor), requires_grad = False)
            batch = batch.squeeze(0)
            context_idx = batch[:,0]
            input_idx = batch[:,1]
            #print(batch.shape)
            tree_path_idx = Variable(tree.lookup_paths(input_idx).type(Tensor), requires_grad = False)
            binary_multipliers = Variable(tree.lookup_binmultipliers(input_idx).type(Tensorf), requires_grad = False)
            #print(tree_path_idx.shape)
            optimE.zero_grad()
            context_vector = embeddings(context_idx)
            prob = tree(context_vector, tree_path_idx,binary_multipliers)
            loss = nll(prob)
            print(idx, totalsize,loss)
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


