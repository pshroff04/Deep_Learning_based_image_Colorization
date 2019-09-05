import numpy as np
from skimage.io import imread
from skimage import color
import random
from skimage.transform import resize
from glob import glob
from os.path import *

# Read into file paths and randomly shuffle them
#________________________CHECK_______________________________________________
root = '*/'
filename_lists = sorted(glob(join(root, '*/*.png')))
random.shuffle(filename_lists)
#############################################################################

#Load points in gamut
points = np.load('pts_in_gamut.npy') #[313, 2]
points = points.astype(np.float64)
points = points[np.newaxis, :, :] #[1, 313, 2]

probs = np.zeros((points.shape[1]), dtype=np.float64) #[313, ]
num = 0

def get_index(in_data):
    """
    Returns a quantised image with the ab color closest in gamut
    """
    expand_in_data = np.expand_dims(in_data, axis=1)
    distance = np.sum(np.square(expand_in_data - points), axis=2)
    return np.argmin(distance, axis=1)


for num, img_f in enumerate(filename_lists):
    img = imread(img_f)
    img = resize(img, (256, 256), preserve_range=True)

    # Make sure the image is rgb format
    if len(img.shape) != 3 or img.shape[2] != 3:
        continue
    img_lab = color.rgb2lab(img) #[H, W, 3]
    img_lab = img_lab.reshape((-1, 3)) #[H*W, 3]
    
    
    img_ab = img_lab[:, 1:].astype(np.float64) #[H*W, 2]

    nd_index = get_index(img_ab)
    for i in nd_index:
        i = int(i)
        probs[i] += 1
    print(num)


# Normalise the probability
probs = probs / np.sum(probs) #[313,]
filename = 'prior_probs.npy'
np.save(filename, probs)