from skimage import color
import numpy as np
import torch

class RGB_to_LAB:

    def __init__(self):
        self.rgb2lab = color.rgb2lab

    def __call__(self, img):
        nd_image = np.array(img)
        lab_image = self.rgb2lab(nd_image).transpose(2,0,1)
        lab_image = torch.from_numpy(lab_image.astype(np.float32))
        return lab_image
