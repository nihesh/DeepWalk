import torch
import torch.nn as nn
from heapq import heapify, heappush, heappop 
import copy
import sys
class Node(nn.Module):
    def __init__(self):
        super(Node, self).__init__()
        self.tensor = None
        self.left = None
        self.right = None
        self.value = None #only when node is leaf
        self.degree = 0 #only for huffman coding
    def __lt__(self, other):
        return self.degree < other.degree
    
class SoftmaxTree(nn.Module):
    def __init__(self, embed_size):
        super(SoftmaxTree, self).__init__()
        self.root = None
        self.codes = {}
        self.embed_size = embed_size
    def generate_codes(self, root, temp_list):
        if(root is None):
            return
        if(root.value is not None):
            self.codes[root.value] = copy.deepcopy(temp_list)
            return
        if(root.left is not None):
            temp_list.append(0)
            self.generate_codes(root.left, temp_list)
            temp_list.pop()
        if(root.right is not None):
            temp_list.append(1)
            self.generate_codes(root.right, temp_list)
            temp_list.pop()
        return 
    def create_huffman(self, deg_list):
        heap = []
        for i in deg_list:
            idx = Node()
            idx.value = i[0]
            idx.degree = i[1]
            heap.append(idx)
            
        heapify(heap)
        while(len(heap) > 1):
            left = heappop(heap)
            right = heappop(heap)
            parent = Node()
            parent.degree = left.degree + right.degree
            parent.left = left
            parent.right = right
            parent.tensor = nn.Parameter(torch.randn(self.embed_size))
            heappush(heap, parent)
        
        root = heappop(heap)
        self.root = root
        self.generate_codes(self.root, [])
    def forward(self, input_node, input_embedding):
        code = self.codes[input_node]
        root = self.root
        prob = 1
        print(code)
        for i in code:
            dotp = torch.dot(input_embedding, root.tensor)
            sigmoid = torch.sigmoid(dotp)
            prob *= sigmoid
            if(i == 0):
                root = root.left
            elif(i == 1):
                root = root.right
            else:
                print("bad binary huffman code for {}, exiting!!!".format(input_node)) 
                sys.exit(-1)  
        return prob          
class EmbeddingMatrix(nn.Module):
    def __init__(self, max_nodes, embed_size):
        super(EmbeddingMatrix, self).__init__()
        self.max_nodes = max_nodes
        self.embed_size = embed_size
        self.matrix = nn.Embedding(self.max_nodes, self.embed_size)
    def forward(self, input):
        return self.matrix(input)