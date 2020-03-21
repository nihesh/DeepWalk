from graph import Graph
from parameters import SoftmaxTree, EmbeddingMatrix
from tqdm import tqdm
import torch
import random
import torch.optim as optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Tensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
import args
import pickle
def nll(prob, eps = args.eps):
    return -1 * torch.log(prob + eps)
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
            print(prob, loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(embedding.parameters(), args.grad_norm)
            torch.nn.utils.clip_grad_norm_(tree.parameters(), args.grad_norm)
            optimE.step()
            optimT.step()
    return tree,embedding,optimT,optimE
            
def deepwalk(tree, embedding, graph, optimT, optimE):
    nodes_index = [i for i in range(graph.num_nodes)]
    for gamma in tqdm(range(1, args.walks_per_vertex+1)):
        random.shuffle(nodes_index)
        for counter,vi in tqdm(enumerate(nodes_index)):
            print("iter {} vertex {}".format(gamma, counter))
            vi_name = graph.idx_to_name_map[vi]
            random_walk = graph.random_walk(vi_name, args.walk_length)
            tree,embedding,optimT,optimE = skipgram(random_walk, graph, tree, embedding, optimT, optimE)
    return tree,embedding,optimT,optimE
if __name__ == "__main__":
    g = Graph("../data/nodes.csv", "../data/edges.csv")
    degreelist = g.get_degrees()
    degreelist = sorted(degreelist, key = lambda x : x[1])
    tree = SoftmaxTree(embed_size = args.embed_size)
    tree.create_huffman(degreelist)
    tree = tree.to(device)
    embeddings = EmbeddingMatrix(max_nodes = args.max_nodes, embed_size = args.embed_size).to(device)
    optimE = optim.Adam(embeddings.parameters(), lr = args.lr)
    optimT = optim.Adam(tree.parameters(), lr = args.lr)
    tree,embeddings,optimT,optimE = deepwalk(tree, embeddings, g, optimT, optimE)
    embeddings = list(embeddings.parameters())[0].data.cpu().numpy()
    
    f = open("embeddings.pkl", "wb")
    pickle.dump(embeddings, f)
    f.close()


