# Author : Nihesh Anderson 
# Date 	 : 21 March, 2020
# File 	 : train.py

ROOT = "./data/"
WALK_LENGTH = 40

import src.dataset as dataset

if(__name__ == "__main__"):

	dataset = dataset.Blog(ROOT, WALK_LENGTH)