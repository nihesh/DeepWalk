import torch
import torch.nn as nn
from heapq import heapify, heappush, heappop 
import copy
import sys
import numpy as np
import args
import math
class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.value = None #stores index of vertex associated with this node
        self.sum_degree = 0 #used in case huffman coding is used
        self.index = None #index of node object assigned in huffman tree, will be filled using inorder traversal
    def __lt__(self, other):
        return self.sum_degree < other.sum_degree
    
class SoftmaxTree(nn.Module):
    def __init__(self, embed_size):
        super(SoftmaxTree, self).__init__()
        self.root = None
        self.codes = {}
        self.embed_size = embed_size
        self.size = 0 #global variable reserved for inorder_traversal
        self.matrix = None #embedding matrix depends on self.size
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
    def inorder_traversal(self, root):
        if(root is None):
            return
        if(root.right is None and root.left is None):
            return
        self.inorder_traversal(root.left)
        root.index = self.size
        self.size += 1
        self.inorder_traversal(root.right)
    def create_huffman(self, deg_list):
        heap = []
        self.size = 0
        for i in deg_list:
            idx = Node()
            idx.value = i[0]
            idx.sum_degree = i[1]
            heap.append(idx)
            
        heapify(heap)
        while(len(heap) > 1):
            left = heappop(heap)
            right = heappop(heap)
            parent = Node()
            parent.sum_degree = left.sum_degree + right.sum_degree
            parent.left = left
            parent.right = right
            heappush(heap, parent)
        
        root = heappop(heap)
        self.root = root
        self.generate_codes(self.root, [])
        self.inorder_traversal(self.root)
        self.matrix = nn.Embedding(self.size, self.embed_size)
        self.matrix.weight.data.copy_(torch.from_numpy(np.random.normal(loc = 0.25, scale = 0.01, size = (self.size, self.embed_size))))
    def create_complete_tree(self, vertex_list):
        """
        creates a complete binary tree
        args: vertex_list
        """
        self.size = 0
        num_vertices = len(vertex_list)
        num_vertices = 2**(math.ceil(math.log(num_vertices, 2)))
        list_vertices = [None for i in range(2*num_vertices)]
        for i in range(num_vertices, num_vertices + len(vertex_list)):
            list_vertices[i] = Node()
            list_vertices[i].value = vertex_list[i - num_vertices]
        for i in range(num_vertices - 1, 0, -1):
            list_vertices[i] = Node()
            list_vertices[i].left = list_vertices[2*i]
            list_vertices[i].right = list_vertices[2*i+1]
        self.root = list_vertices[1]
        self.generate_codes(self.root, [])
        self.inorder_traversal(self.root)
        self.matrix = nn.Embedding(self.size, self.embed_size)
        self.matrix.weight.data.copy_(torch.from_numpy(np.random.normal(loc = 0.25, scale = 0.15, size = (self.size, self.embed_size))))  
    
    def get_path(self, input_node_idx):
        code = self.codes[input_node_idx]
        root = self.root
        ret = []
        mul = []
        for i in code:
            ret.append(root.index)
            if(i == 0):
                root = root.left
                mul.append(1)
            else:
                root = root.right
                mul.append(-1)
        return ret,mul
    def forward(self, context_embedding, input_path_idxs, binary_multiplier):
        input_vectors = self.matrix(input_path_idxs)
        context_embedding = context_embedding.unsqueeze(-1)
        probs = torch.matmul(input_vectors, context_embedding)
        probs = probs.squeeze(-1)
        probs = probs * binary_multiplier
        probs = torch.sigmoid(probs)
        prob = torch.prod(probs, dim = 1)
        return prob
class EmbeddingMatrix(nn.Module):
    def __init__(self, num_nodes, embed_size):
        super(EmbeddingMatrix, self).__init__()
        self.num_nodes = num_nodes
        self.embed_size = embed_size
        self.matrix = nn.Embedding(self.num_nodes, self.embed_size)
        self.matrix.weight.data.copy_(torch.from_numpy(np.random.uniform(low = args.low_weight/self.embed_size, high = args.high_weight/self.embed_size, size = (self.num_nodes, self.embed_size))))
    
    def forward(self, input):
        return self.matrix(input)