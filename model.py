from __future__ import print_function
from torch.autograd import Variable
import torch
import torch.nn as nn
from layers import SoftEncodingLayer, NonGrayMaskLayer, ReweighingLayer
import numpy as np

def weights_init(model):
    if type(model) in [nn.Conv2d, nn.Linear]:
        nn.init.xavier_normal(model.weight.data)
        nn.init.constant(model.bias.data, 0.1)
        
        
class ColorizationNetwork_L(nn.Module):
    def __init__(self):
        super(ColorizationNetwork_L, self).__init__()
        self.features = nn.Sequential(
                
            # conv1
            nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(), #[batch, 64, 224, 224]
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU(), #[batch, 64, 112, 112]
            nn.BatchNorm2d(num_features = 64),
            
            # conv2
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(), #[batch, 128, 112, 112]
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU(), #[batch, 128, 56, 56]
            nn.BatchNorm2d(num_features = 128),
            
            # conv3
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(), #[batch, 256, 56, 56]
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(), #[batch, 256, 56, 56]
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU(), #[batch, 256, 28, 28]
            nn.BatchNorm2d(num_features = 256),
            
            # conv4
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(), #[batch, 512, 28, 28]
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(), #[batch, 512, 28, 28]
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(), #[batch, 512, 28, 28]
            nn.BatchNorm2d(num_features = 512),
            
            # conv5
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 2, dilation = 2),
            nn.ReLU(), #[batch, 512, 28, 28]
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 2, dilation = 2),
            nn.ReLU(), #[batch, 512, 28, 28]
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 2, dilation = 2),
            nn.ReLU(), #[batch, 512, 28, 28]
            nn.BatchNorm2d(num_features = 512),
            
            # conv6
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 2, dilation = 2),
            nn.ReLU(), #[batch, 512, 28, 28]
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 2, dilation = 2),
            nn.ReLU(), #[batch, 512, 28, 28]
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 2, dilation = 2),
            nn.ReLU(), #[batch, 512, 28, 28]
            nn.BatchNorm2d(num_features = 512),
            
            # conv7
            nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = 3, stride = 1, padding = 1, dilation = 1),
            nn.ReLU(), #[batch, 256, 28, 28]
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1, dilation = 1),
            nn.ReLU(), #[batch, 256, 28, 28]
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1, dilation = 1),
            nn.ReLU(), #[batch, 256, 28, 28]
            nn.BatchNorm2d(num_features = 256),
            
            # conv8
            nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 4, stride = 2, padding = 1, dilation = 1),
            nn.ReLU(), #[batch, 128, 56, 56]
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, dilation = 1),
            nn.ReLU(), #[batch, 128, 56, 56]
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1, dilation = 1),
            nn.ReLU(), #[batch, 128, 56, 56]
            
            # conv8_313
            nn.Conv2d(in_channels = 128, out_channels = 313, kernel_size = 1, stride = 1,dilation = 1)
            #[batch, 313, 56, 56]
        )
        self.apply(weights_init)

    def forward(self, img_L):
        Z_pred = self.features(img_L) #[batch, 313, 56, 56]
        return Z_pred
    


class ColorizationNetwork(nn.Module):
    """
    This class represents the Colorisation Network
    """
    def __init__(self, batchNorm=True, pretrained=False):
        super(ColorizationNetwork, self).__init__()
        
        """
        self.nnecnclayer = NNEncLayer()
        self.priorboostlayer = PriorBoostLayer()
        self.nongraymasklayer = NonGrayMaskLayer()
        # self.rebalancelayer = ClassRebalanceMultLayer()
        self.rebalancelayer = Rebalance_Op.apply
        # Rebalance_Op.apply
        self.pool = nn.AvgPool2d(4,4)
        self.upsample = nn.Upsample(scale_factor=4)
        self.bw_conv = nn.Conv2d(1,64,3, padding=1)
        self.main = VGG(make_layers(cfg, batch_norm=batchNorm))
        self.main.classifier = nn.ConvTranspose2d(512,256,4,2, padding=1)
        self.relu = nn.ReLU()
        self.conv_8 = conv(256,256,2,[1,1], batchNorm=False)
        self.conv313 = nn.Conv2d(256,313,1,1)
        """
        
        #Components required to process L component
        self.ColorizationNetwork_L = ColorizationNetwork_L()
        self.upsample = nn.Upsample(scale_factor = 4)
        
        #Components required to process ab component
        self.pool = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)
        self.soft_encoding_layer = SoftEncodingLayer()
        self.non_gray_mask_layer = NonGrayMaskLayer()
        self.reweighting_layer = ReweighingLayer()
        
    def forward(self, img):
        
        #Processing the L component
        #gt_img_l = (gt_img[:,:1,:,:] - 50.) * 0.02
        #print('Input Shape', img.shape)
        img_L = img[:, :1, :, :] #[batch, 1, 224, 224]
        Z_pred = self.ColorizationNetwork_L(img_L) #[batch, 313, 56, 56]
        
        # Processing the ab component 
        img_ab = img[:,1:,:,:] # [batch, 2, 224, 224]
        img_ab_downsample = self.pool(img_ab).cpu().data.numpy() # [batch, 2, 56, 56] #numpy
        
        #groundtruth Z
        img_ab_prob_dist = self.soft_encoding_layer.evaluate(img_ab_downsample) # [batch, 313, 56, 56]
        img_ab_prob_dist_argmax = np.argmax(img_ab_prob_dist, axis = 1).astype(np.int32)
        nongray_flag = self.non_gray_mask_layer.evaluate(img_ab_downsample) #[batch, 1, 1, 1]
        
        #Weight for class rebalancing 
        weight_per_pixel = self.reweighting_layer.evaluate(img_ab_prob_dist) # [batch, 1, 56, 56]
        
        #Final Weight per pixel
        weight_per_pixel_mask = (weight_per_pixel * nongray_flag).astype('float32') #[batch, 1, 56, 56]
        
        #Convert to tensors, ALL MUST BE float32 type
        weights = Variable(torch.from_numpy(weight_per_pixel_mask)).cuda()
        Z_groundtruth_argmax = Variable(torch.from_numpy(img_ab_prob_dist_argmax))
        Z_groundtruth_argmax = Z_groundtruth_argmax.type(torch.LongTensor).cuda()
        #Z_pred
        
        #Return is different for train and test mode
        if self.training:
            return weights, Z_groundtruth_argmax, Z_pred
        else:
            return weights, Z_groundtruth_argmax, Z_pred, self.upsample(Z_pred)
        
