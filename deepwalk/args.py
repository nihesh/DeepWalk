window_length = 3
walks_per_vertex = 300
walk_length = 10
embed_size = 2
max_nodes = 34
lr = 0.01
grad_norm = 5
hdf5file = "walk_pairs.hdf5"
low_weight = -0.5
high_weight = 0.5 
tree_constructor = "complete" #choose from complete or huffman
use_old_copy = True #if true, doesnt create new walk_pairs.hdf5
#NOTE - using huffman or complete tree coding creates entirely different trees and hence different walk_pairs.hdf5 files
#this means that walk_pairs of huffman coded tree cant be used for complete tree
#Further, huffman support batch size 1 where as complete supports any batch size
batch_size = 64 if tree_constructor == "complete" else 1