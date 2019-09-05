import numpy as np


lmda = 0.5

variable_path = 'prior_probs.npy'

# empirical prior probability
prior_probs = np.load(variable_path)

# define uniform probability
uniform_probs = np.zeros_like(prior_probs)
uniform_probs[prior_probs!=0] = 1.
uniform_probs = uniform_probs/np.sum(uniform_probs)

#mixed prior_probs
prior_mix = (1-lmda)*prior_probs + lmda*uniform_probs

# Set the weights
weights = prior_mix**(-1)

# re-normalize
weights = weights/np.sum(prior_probs*weights)  #[313, ]

filename = 'weight_ab.npy'
np.save(filename, weights)

