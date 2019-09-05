import numpy as np
import warnings
import os
import sklearn.neighbors as neighbors
import torch
from skimage import color
from torch.autograd import Function
import pickle

class SoftEncodingLayer():
    """
    Converts ab values to probability distribution for the colors in gamut
    """
    def __init__(self):
        """
        Q : size of gamut
        model_loc : location of Nearest Neighbor model
        NN_model : Nearest Neighbor model
        NN : Number of Nearest Neighbors returned by NN_model
        """
        self.Q = 313 
        self.model_path = './prior_prob/nbrs.pkl'
        self.NN_model = pickle.load(open(self.model_path, 'rb'))
        self.NN = self.NN_model.get_params()['n_neighbors'] 
        self.sigma = 5.0
        
    def evaluate(self, x):
        """
        Input : 
            x: Downsampled ab channel, nparray shape : [batch, 2, 56, 56]
        Output :
            x_prob_dist : Prob distribution over the ab channel, nparray shape : [batch, 313, 56, 56]
        """
        #Prepare
        flat_x = flatten_nd_array(x, axis=1) #[batch*56*56, 2]
        n_points = flat_x.shape[0] #[batch*56*56]
        flat_x_prob_dist = np.zeros([n_points, self.Q]) #[batch*56*56, 313]
        row_indices =  np.arange(0, n_points, dtype='int')[:,np.newaxis] #[batch*56*56, 1]
        
        #Find the Nearest Neighbors
        (dists, col_indices) = self.NN_model.kneighbors(flat_x) #[batch*56*56, NN] for both
        
        #Smooth using Gaussian Kernel
        weights = np.exp(-dists**2/(2*self.sigma**2)) #[batch*56*56, NN]
        weights = weights/(np.sum(weights,axis=1)[:,np.newaxis]) #[batch*56*56, NN]
        
        #Assign the values
        flat_x_prob_dist[row_indices, col_indices] = weights #[batch*56*56, 313]
         
        #Reshape and set dtype
        x_prob_dist = unflatten_2d_array(flat_x_prob_dist, x, axis=1) #[batch, 313, 56, 56]
        x_prob_dist = x_prob_dist.astype('float32') #As we need tensor of type float32 later
        
        return x_prob_dist
    
class NonGrayMaskLayer():
    """
    Returns 1 if image is not GRAYSCALE else 0
    """
    def __init__(self):
        self.thresh = 5 # threshold on ab value

    def evaluate(self, x):
        """
        Input : 
            Downsampled ab channel, nparray shape : [batch, 2, 56, 56]
        Output :
            Numpy array of size [batch, 1, 1, 1]
                1 if image is not GRAYSCALE 
                0 if image is GRAYSCALE
        """
        #if len(x) == 0:
         #   raise Exception("NonGrayMaskLayer should have inputs")
         
        # if an image has any (a,b) value which exceeds threshold, output 1
        return (np.sum(np.sum(np.sum(np.abs(x) > self.thresh,axis=1),axis=1),axis=1) > 0)[:,np.newaxis,np.newaxis,np.newaxis]

class ReweighingLayer():
    """
    Evaluates the weights for each pixel as required in the loss function
    """
    def __init__(self):
        self.variable_path = './prior_prob/weight_ab.npy'
        self.weight_ab = np.load(self.variable_path) # [313,]
    
    def evaluate(self, x):
        """
        Input: 
            ab_prob_dist , shape : [batch, 313, 56, 56]
        Output:
            A weight for each pixel, shape : [batch, 1, 56, 56]
        """
        x_argmax = np.argmax(x, axis=1) # [batch, 56, 56]
        weights_per_pixel = self.weight_ab[x_argmax] #[batch, 56, 56]
        
        return weights_per_pixel[:,np.newaxis,:] #[batch, 1, 56, 56]
        

def flatten_nd_array(pts_nd,axis=1):
    ''' Flatten an nd array into a 2d array with a certain axis
    INPUTS
        pts_nd       N0xN1x...xNd array
        axis         integer
    OUTPUTS
        pts_flt     prod(N \ N_axis) x N_axis array     '''
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0,NDIM),np.array((axis))) # non axis indices
    NPTS = np.prod(SHP[nax])
    axorder = np.concatenate((nax,np.array(axis).flatten()),axis=0)
    pts_flt = pts_nd.transpose((axorder))
    pts_flt = pts_flt.reshape(NPTS,SHP[axis])
    return pts_flt

def unflatten_2d_array(pts_flt,pts_nd,axis=1,squeeze=False):
    ''' Unflatten a 2d array with a certain axis
    INPUTS
        pts_flt     prod(N \ N_axis) x M array
        pts_nd      N0xN1x...xNd array
        axis        integer
        squeeze     bool     if true, M=1, squeeze it out
    OUTPUTS
        pts_out     N0xN1x...xNd array        '''
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0,NDIM),np.array((axis))) # non axis indices
    NPTS = np.prod(SHP[nax])

    if(squeeze):
        axorder = nax
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)
    else:
        axorder = np.concatenate((nax,np.array(axis).flatten()),axis=0)
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        NEW_SHP.append(M)
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)

    return pts_out
        
        
        
        
