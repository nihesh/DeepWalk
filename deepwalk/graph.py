import args
import random
class Graph:
    def __init__(self, node_file, edge_file):
        self.adj, self.idx_to_name_map, self.name_to_idx_map = self.read_file(node_file, edge_file)
        self.num_nodes = len(self.idx_to_name_map)
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
        
    def read_file(self, node_file, edge_file):
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
        return adj, idx_to_name, name_to_idx
    def random_walk(self, vertex_start, walk_length):
        """
        args:
        vertex_start = id of start vertex (integer)
        walk_length = length of random walk (integer)
        """
        ret = []
        for i in range(walk_length):
            ret.append(vertex_start)
            if((len(self.adj[vertex_start])) == 0):
                break
            next_idx = random.randint(0, len(self.adj[vertex_start])-1)
            next_node = self.adj[vertex_start][next_idx]
            vertex_start = next_node
        return ret
from torch.utils.data import Dataset
import h5py
class WalkPairData(Dataset):
    def __init__(self, hdf5_path):
        super(WalkPairData, self).__init__()
        self.hdf5_path = hdf5_path
        self.f = h5py.File(self.hdf5_path, "r")
    def __len__(self):
        return len(self.f)
    def __getitem__(self, idx):
        return self.f[str(idx)].value