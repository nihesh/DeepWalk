import pickle
import numpy as np
import matplotlib.pyplot as plt
f = open("embeddings.pkl", "rb")
mat_new,mat_old = pickle.load(f)

def draw(mat):
    text = [i for i in range(1,mat.shape[0]+1)]
    fig, ax = plt.subplots()
    ax.scatter(mat[:,0], mat[:,1])
    for i, txt in enumerate(text):
        ax.annotate(txt, (mat[:,0][i], mat[:,1][i]))
    plt.show()
    plt.close()

draw(mat_old)
draw(mat_new)
