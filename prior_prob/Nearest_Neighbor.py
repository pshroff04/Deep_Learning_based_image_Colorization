import numpy as np
import sklearn.neighbors as neighbors
import pickle

variable_path = 'pts_in_gamut.npy'
pts_in_gamut = np.load(variable_path)
pts_in_gamut = pts_in_gamut.astype(np.float64) #[313, 2]

NN = 5
nbrs = neighbors.NearestNeighbors(n_neighbors=NN, algorithm='ball_tree').fit(pts_in_gamut)

filename = 'nbrs.pkl'
pickle.dump(nbrs, open(filename, 'wb'))