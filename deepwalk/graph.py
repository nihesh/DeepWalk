import args
import random
import numpy as np
import pickle
import sys
class Graph:
    def __init__(self, node_file, edge_file, subset_size = -1):
        self.adj, self.idx_to_name_map, self.name_to_idx_map, self.transformed_map = self.read_file(node_file, edge_file, subset_size)
        self.num_nodes = len(self.idx_to_name_map)
    def save_graph(self, path):
        f = open(path, "wb")
        pickle.dump([self.adj,self.idx_to_name_map], f)
        f.close() 
    def get_degrees(self):
        """
        returns a list of tuples (id,len) where id is index of vertex and len is degree of this vertex
        """
        ret = []
        for k,v in self.adj.items():
            ret.append((k,len(v)))
        return ret
    def get_vertices(self):
        return list(self.adj.keys())
    def subset_graph(self, adj, idx_to_name, name_to_idx, subset_size):
        #subset nodes uniformly at random fromt the original graph
        total_nodes = len(idx_to_name)
        selected_idx = np.sort(np.random.choice(total_nodes, subset_size, replace = False))
        transformed_idx = [i for i in range(subset_size)]
        transformed_map = {selected_idx[i] : transformed_idx[i] for i in range(subset_size)}
        
        selected_idx = {i:1 for i in selected_idx}
        idx_to_name = {transformed_map[i]:idx_to_name[i] for i in list(idx_to_name.keys()) if i in selected_idx}
        name_to_idx = {idx_to_name[i]:i for i in list(idx_to_name.keys())}
        new_adj = {}
        for k,v in adj.items():
            if(selected_idx.get(k,-1) == -1):
                continue
            new_list = []
            for id in v:
                if(selected_idx.get(id,-1) == -1):
                    continue
                new_list.append(transformed_map[id])
            new_adj[transformed_map[k]] = new_list
        
        return new_adj,idx_to_name,name_to_idx,transformed_map
    def read_file(self, node_file, edge_file, subset_size):
        """
        args:
        node_file: path to file of vertices
        edge_file: path to edge file
        returns:
        adj: adjacency matrix, a dict with key as vertex id and value as list of vertex ids
        idx_to_name: dict with idx to vertex name mapping
        name_to_idx: dict with vertex name to idx mapping
        """
        node_file = open(node_file, 'r')
        edge_file = open(edge_file, 'r')
        adj = {}
        idx_to_name = {}
        name_to_idx = {}
        for idx, line in enumerate(node_file.readlines()):
            line = line.strip()
            adj[idx] = []
            idx_to_name[idx] = line
            name_to_idx[line] = idx
        for line in edge_file.readlines():
            line = line.strip().split(",")
            node1 = line[0].strip()
            node2 = line[1].strip()
            id_node1 = name_to_idx[node1]
            id_node2 = name_to_idx[node2]
            adj[id_node1].append(id_node2)
            adj[id_node2].append(id_node1)
        transformed_map = None
        if subset_size != -1:
            adj, idx_to_name, name_to_idx,transformed_map = self.subset_graph(adj, idx_to_name, name_to_idx, subset_size)
        return adj, idx_to_name, name_to_idx,transformed_map
    def random_walk(self, vertex_start, walk_length):
        """
        args:
        vertex_start = id of start vertex (integer)
        walk_length = length of random walk (integer)
        """
        ret = []
        if(self.adj.get(vertex_start, -1) == -1):
            return ret
        for i in range(walk_length):
            ret.append(vertex_start)
            if((len(self.adj[vertex_start])) == 0):
                break
            next_idx = random.randint(0, len(self.adj[vertex_start])-1)
            next_node = self.adj[vertex_start][next_idx]
            vertex_start = next_node
        return ret
