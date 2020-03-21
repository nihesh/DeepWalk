import args
import random
class Graph:
    def __init__(self, node_file, edge_file):
        self.adj, self.idx_to_name_map, self.name_to_idx_map = self.read_file(node_file, edge_file)
        self.num_nodes = len(self.idx_to_name_map)
    def get_degrees(self):
        ret = []
        for k,v in self.adj.items():
            ret.append((k,len(v)))
        return ret
    def read_file(self, node_file, edge_file):
        node_file = open(node_file, 'r')
        edge_file = open(edge_file, 'r')
        adj = {}
        idx_to_name = {}
        name_to_idx = {}
        for idx, line in enumerate(node_file.readlines()):
            line = line.strip()
            adj[line] = []
            idx_to_name[idx] = line
            name_to_idx[line] = idx
        for line in edge_file.readlines():
            line = line.strip().split(",")
            node1 = line[0]
            node2 = line[1]
            adj[node1].append(node2)
            adj[node2].append(node1)
        return adj, idx_to_name, name_to_idx
    def random_walk(self, vertex_start, walk_length):
        if(walk_length == 0):
            return []
        if((len(self.adj[vertex_start])) == 0):
            return []
        next_idx = random.randint(0, len(self.adj[vertex_start])-1)
        next_node = self.adj[vertex_start][next_idx]
        return [vertex_start] + self.random_walk(next_node, walk_length-1)
    