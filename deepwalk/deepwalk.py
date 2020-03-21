from graph import Graph
from parameters import SoftmaxTree, EmbeddingMatrix
import torch
if __name__ == "__main__":
    g = Graph("../data/nodes.csv", "../data/edges.csv")
    degreelist = g.get_degrees()
    degreelist = sorted(degreelist, key = lambda x : x[1])
    tree = SoftmaxTree(embed_size = 16)
    tree.create_huffman(degreelist)
    embeddings = EmbeddingMatrix(max_nodes = 11000, embed_size = 16)
    
    
    index = 512
    indexname = g.idx_to_name_map[index]
    embedding = embeddings(torch.LongTensor([index])).squeeze(0)
    print(embedding)
    prob = tree(indexname, embedding)
    print(prob)