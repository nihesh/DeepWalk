window_length = 10
walks_per_vertex = 80
walk_length = 40
embed_size = 32
lr = 0.025
grad_norm = 5
low_weight = -0.5
high_weight = 0.5 
tree_constructor = "complete" #choose from complete or huffman
use_old_copy = False #if true, doesnt create new walk_pairs.hdf5
batch_size = 1 if tree_constructor == "complete" else 1
subset_size = -1
