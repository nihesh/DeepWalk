from graph import Graph
from parameters import SoftmaxTree, EmbeddingMatrix
import torch
import random
import torch.optim as optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Tensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
import args
import pickle
def nll(prob, eps = 1e-20):
    return -1 * torch.log(prob)
def skipgram(random_walk, graph, tree, embedding, optimT, optimE):
    for idx in range(len(random_walk)):
        for j in range(max(0, idx-args.window_length), min(len(random_walk), idx + args.window_length + 1)):
            input_node_name = random_walk[j]
            index = (graph.name_to_idx_map[random_walk[idx]])
            context_embedding = embedding(Tensor([index])).squeeze(0)
            prob = tree(input_node_name, context_embedding)
            optimE.zero_grad()
            optimT.zero_grad()
            loss = nll(prob)
            #print(loss)
            loss.backward()
            optimE.step()
            optimT.step()
    return tree,embedding,optimT,optimE
            
def deepwalk(tree, embedding, graph, optimT, optimE):
    nodes_index = [i for i in range(graph.num_nodes)]
    for gamma in range(1, args.walks_per_vertex+1):
        random.shuffle(nodes_index)
        for counter,vi in enumerate(nodes_index):
            print("iter {} vertex {}".format(gamma, counter))
            vi_name = graph.idx_to_name_map[vi]
            random_walk = graph.random_walk(vi_name, args.walk_length)
            tree,embedding,optimT,optimE = skipgram(random_walk, graph, tree, embedding, optimT, optimE)
            break
        break
    return tree,embedding,optimT,optimE
if __name__ == "__main__":
    g = Graph("../data/nodes.csv", "../data/edges.csv")
    degreelist = g.get_degrees()
    degreelist = sorted(degreelist, key = lambda x : x[1])
    tree = SoftmaxTree(embed_size = args.embed_size).to(device)
    tree.create_huffman(degreelist)
    embeddings = EmbeddingMatrix(max_nodes = args.max_nodes, embed_size = args.embed_size).to(device)
    optimE = optim.Adam(embeddings.parameters(), lr = args.lr)
    optimT = optim.Adam(tree.parameters(), lr = args.lr)
    tree,embeddings,optimT,optimE = deepwalk(tree, embeddings, g, optimT, optimE)
    embeddings = list(embeddings.parameters())[0].data.numpy()
    
    f = open("embeddings.pkl", "wb")
    pickle.dump(embeddings, f)
    f.close()

    
