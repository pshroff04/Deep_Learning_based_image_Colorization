import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from model_vgg import ColorizationNetwork
from skimage.color import lab2rgb
from skimage import io
import os
from tensorboardX import SummaryWriter
gamut = np.load('./prior_prob/pts_in_gamut.npy')


class training:

    def __init__(self, args):
        self.model = ColorizationNetwork()
        self.batch_size = args.batch_size
        self.val_batch_size = args.val_batch_size
        self.num_iterations = args.num_iteration
        self.gpu = args.gpu
        self.pretrained = args.pretrained
        self.epoch = args.epoch
        self.save_directory = args.save_directory
        self.resume = args.resume
        self.lr = args.lr
        self.lr_update_iter = args.lr_update_iter
        self.loss_arr,self.test_arr = [],[]
        #define criterion
        self.criterion = nn.CrossEntropyLoss(reduce=False).cuda()
        #optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-3)
        self.logger = SummaryWriter('./Tensorboard_logs')
        self.weight_dir = args.weight_dir

    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def save_imgs(self, tensor, filename):
        for index, im in enumerate(tensor):
            # print(im.shape)
            # im =np.clip(im.numpy().transpose(1,2,0), -1, 1) 
            img_rgb_out = (255*np.clip(lab2rgb(im),0,1)).astype('uint8')
            io.imsave(filename +'rgb'+ str(index) + '.png', img_rgb_out )

    def restore(self,resume_iter):
        print('Restoring model stage from ./net_%d.pth'%(resume_iter))
        if self.weight_dir:
            path = os.path.join(self.weight_dir, '/net_%d.pth'%(resume_iter))
        else:
            path = os.path.join(self.save_directory, '/net_%d.pth'%(resume_iter))
        
        checkpoint_dict = torch.load(path)
        keys = list(checkpoint_dict['state_dict'].keys())
        for key in keys:
            checkpoint_dict['state_dict'][key[7:]] =  checkpoint_dict['state_dict'][key]
            del checkpoint_dict['state_dict'][key]
        self.model.load_state_dict(checkpoint_dict['state_dict'])
        self.lr = checkpoint_dict['lr']
        self.loss_arr = checkpoint_dict['train_loss_list']
        self.test_arr = checkpoint_dict['val_loss_list']
        return checkpoint_dict['iteration']


    def train(self, train_loader, val_loader):

        start_iter = 0
        if self.resume:
            # TODO
            start_iter = self.restore(self.resume)
        #transfer to GPU
        
        model = nn.DataParallel(self.model).cuda()

        #Set to training mode
        model.train()

        
        
        print('..........................Starting training.................')

        #start training
        for epoch in range(self.epoch):
            count = 0
            for i, data in enumerate(train_loader, start_iter):
                img, _ = data
                
                img = Variable(img).cuda()
                weights, Z_gt, Z_pred = model(img)
                batch_size = weights.shape[0]
                h = weights.shape[2]
                w = weights.shape[3]
                loss = torch.sum((self.criterion(Z_pred, Z_gt))*(weights.squeeze(dim = 1)))/(batch_size*1.0*h*w)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
                #logging
                if (i+1) % 100 == 0:
                    print('At Epoch %d iteration %d, train loss is %f'%(epoch,i,loss.data[0]))
                    self.loss_arr.append(loss.data[0])
                    self.logger.add_scalar('train_loss', loss.data[0], i+1)
    
                #Update learning rate if required
                if (i+1) % self.lr_update_iter == 0:
                    self.record_iters = i
                    if self.lr > 1e-8:
                        self.lr *= 0.316
                    self.update_lr(self.lr)
    
                #validate/Test
                if (i+1)%2000 ==0:
                    self.test(val_loader, i)
                    model.train()
    
    
                #checkpoint
                if (i+1)%2000 == 0:
                    state_dict = model.state_dict()
                    checkpoint = {'iteration': i,
                                  'state_dict': state_dict,
                                  'lr' : self.lr,
                                  'train_loss_list':self.loss_arr,
                                  'val_loss_list': self.test_arr}
                    save_path = os.path.join(self.save_directory, './net_%d.pth'%(i+1))
                    torch.save(checkpoint, save_path)
            
                if count > self.num_iterations:
                    break
                count +=1   
            print('...............Training Completed...........')
        
        
    def test(self, test_loader, curr_iter, inference_iter=0):
        
        # Load the trained generator.
        self.optimizer.zero_grad()
        data_iter = iter(test_loader)

        if inference_iter:
            self.restore(inference_iter)

        if inference_iter:
            print('Start inferencing....................')
        else:
            print('Start Validating.....................')
            
        
        self.model.eval()  # Set g_model to training mode

        img_dir =  os.path.join(self.save_directory, 'Test','%d/' % (curr_iter))
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        len_record = len(test_loader)
        softmax_op = torch.nn.Softmax(dim = 1)
        test_loss = 0.0

        for global_iteration in range(len_record):
            
            
            #print('completed %d of %d' % (global_iteration, len_record))
            
            # Iterate over data.
            img , _ = next(data_iter)

            # wrap them in Variable
            img = Variable(img.cuda(), volatile=True)

            weights, Z_gt, Z_pred, Z_pred_upsample  = self.model(img)
            batch_size = weights.shape[0]
            h = weights.shape[2]
            w = weights.shape[3]
            loss = torch.sum((self.criterion(Z_pred, Z_gt)*weights.squeeze(dim = 1)))/(batch_size*1.0*h*w)
            test_loss += loss.data[0]

            img_L = img[:,:1,:,:] #[batch, 1, 224, 224]

            # post-process
            Z_pred_upsample *= 2.606
            Z_pred_upsample = softmax_op(Z_pred_upsample).cpu().data.numpy()

            fac_a = gamut[:,0][np.newaxis,:,np.newaxis,np.newaxis]
            fac_b = gamut[:,1][np.newaxis,:,np.newaxis,np.newaxis]

            img_L = img_L.cpu().data.numpy().transpose(0,2,3,1) #[batch, 224, 224, 1]
            frs_pred_ab = np.concatenate((np.sum(Z_pred_upsample * fac_a, axis=1, keepdims=True), np.sum(Z_pred_upsample * fac_b, axis=1, keepdims=True)), axis=1).transpose(0,2,3,1)
            #[batch, 224, 224, 2]
            
            frs_predic_imgs = np.concatenate((img_L, frs_pred_ab ), axis = 3) #[batch, 224, 224, 3]
            #print('Saving image %s%d_frspredic_' %  (img_dir, global_iteration))
            self.save_imgs(frs_predic_imgs, '%s%d_frspredic_' %  (img_dir, global_iteration))
        test_loss = test_loss/float(len_record)
        print('val loss is %f'%(test_loss))
        self.test_arr.append(test_loss)
        self.logger.add_scalar('val_loss', test_loss, curr_iter)
        print('Finished Validating.....................')
