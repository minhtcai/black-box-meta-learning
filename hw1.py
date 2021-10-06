import argparse
import os
import torch

import torch.nn.functional as F
import numpy as np
from torch import nn
from load_data import DataGenerator
from dnc import DNC
from google_drive_downloader import GoogleDriveDownloader as gdd
from torch.utils.tensorboard import SummaryWriter


class MANN(nn.Module):

    def __init__(self, num_classes, samples_per_class, model_size=128, input_size=784, use_dnc=False):
        super(MANN, self).__init__()
        
        def initialize_weights(model):
            nn.init.xavier_uniform_(model.weight_ih_l0)
            nn.init.zeros_(model.bias_hh_l0)
            nn.init.zeros_(model.bias_ih_l0)
    
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.input_size = input_size
        self.layer1 = torch.nn.LSTM(num_classes + input_size, 
                                    model_size, 
                                    batch_first=True)
        self.layer2 = torch.nn.LSTM(model_size,
                                    num_classes,
                                    batch_first=True)
        initialize_weights(self.layer1)
        initialize_weights(self.layer2)
        
        self.use_dnc = use_dnc
        self.dnc = DNC(
                       input_size=num_classes + input_size,
                       output_size=num_classes,
                       hidden_size=model_size,
                       rnn_type='lstm',
                       num_layers=1,
                       num_hidden_layers=1,
                       nr_cells=num_classes,
                       cell_size=64,
                       read_heads=1,
                       batch_first=True,
                       gpu_id=0,
                       ).to(torch.float)

    def forward(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: tensor
                A tensor of shape [B, K+1, N, 784] of flattened images
            
            labels: tensor:
                A tensor of shape [B, K+1, N, N] of ground truth labels
        Returns:
            
            out: tensor
            A tensor of shape [B, K+1, N, N] of class predictions
        """
        #############################
        #### YOUR CODE GOES HERE ####
        #############################

        # SOLUTION:
        N = self.num_classes
        K = self.samples_per_class
        dim = self.input_size
        B = len(input_images)
        
        # reshape tensors to process [B * (K+1) * N, dim] and [B * (K+1) * N, N]
        reshaped_image = torch.reshape(input_images, (B * N * (K+1), dim))
        reshaped_label = torch.reshape(input_labels, (B * N * (K+1), N))
        
        # conatenate query image with label of zeros
        # Notes: in the previous part, we put all the query set of each character in each batch to the bottom, for each batch we have (1 * N) at the
        # bottom of input [B, K + 1, N, 784] are the query set
        # E.g: n = 4 classes, k = 10 ways, batch = 2, we have output shape of image batch is (2, 11, 4, 784)
        # For first batch, if we resize the tensor to (1, 44, 784), the last three vectors of index 41, 42, 43, 44 are in the query set
        # And if we resize tensor with both batch to (2 * 44, 784) = (88, 784) the query set is in position (th) (41, 42, 43, 44) and (85, 86, 87, 88)
        # That are the indexes that we need to concatenate with vector of zeros, we can start to change the labels of these indexes before concatenating
        for i in range(len(reshaped_label)):
            # find last set of batch label, for e.g. example above, index should be 40, 41, 42, 43 and 84, 85, 86, 87
            if (i + 1) % (N * (K + 1)) == 0:
                for j in range(i, i - N, -1):
                    pass
                    # get problem with mutability # https://discuss.pytorch.org/t/why-change-a-tensors-value-in-a-function-can-change-the-outer-tensor/94661
                    # https://deep-learning-phd-wiki.readthedocs.io/en/latest/src/code/learnPytorch.html
                    #reshaped_label[j] = torch.zeros(N)

        # Since Torch Tensor is an immutable object, so in this case I convert it to numpy tensor to set query set labels to vector of zeros
        img_test = np.asarray(reshaped_image.cpu())
        label_test = np.asarray(reshaped_label.cpu())
        for i in range(len(label_test)):
            if (i + 1) % (N * (K + 1)) == 0:
                for j in range(i, i - N, -1):
                    label_test[j] = np.zeros(N)
        
        # Concatenate image and label vector and reshapre back to [B, (K+1)*N, dim + N]
        cat_test = np.concatenate((img_test, label_test), axis = 1)
        cat_test = np.reshape(cat_test, (B, (K+1) * N, dim + N))
        cat_test = torch.from_numpy(cat_test)
        cat_test = cat_test.to(torch.double)
        cat_test = cat_test.to('cuda')

        # Pass data to model
        if self.use_dnc == True:
        	output, _ = self.dnc(cat_test.float())
        	return output
        else:
	        # pass data through model and reshape output the [B, K + 1, N, N]
	        output, _ = self.layer1(cat_test)
	        output, _ = self.layer2(output)
	        output = torch.reshape(output, (B, K + 1, N, N))
        
       		return output.to(torch.float)


    def loss_function(self, preds, labels):
        """
        Computes MANN loss
        Args:
            preds: tensor
                A tensor of shape [B, K+1, N, N] of network outputs
            
            labels: tensor
                A tensor of shape [B, K+1, N, N] of class labels
                
        Returns:
            scalar loss
        """
        #############################
        #### YOUR CODE GOES HERE ####
        #############################

        # SOLUTION: 
        
        N = self.num_classes
        K = self.samples_per_class
        dim = self.input_size
        B = len(labels)
        
        # Reshape two inputs into [B * (K+1) * N, N]
        reshaped_preds = torch.reshape(preds, (B * (K+1) * N, N))
        reshaped_labels = torch.reshape(labels, (B * (K+1) * N, N))
        
        # Get prediction and label from the last items, should be  last N * 1 sample 
        preds_N = reshaped_preds[(len(reshaped_preds) -  B*N):, :]
        labels_N = reshaped_labels[(len(reshaped_preds) -  B*N):, :]
        
        # one-hot encoding for the label [B*N, N] and [B*N,]
        # transfor to 8 * 1
        labels_N = torch.argmax(labels_N, dim = 1)
        
        preds_N = preds_N.to('cuda')
        
        output = F.cross_entropy(preds_N, labels_N)
        
        return output    
        



def train_step(images, labels, model, optim):
    labels = labels.to('cuda')
    images = images.to('cuda')
    predictions = model(images, labels)
    loss = model.loss_function(predictions, labels)
    
    optim.zero_grad()
    loss.backward()
    optim.step()
    return predictions.detach(), loss.detach()


def model_eval(images, labels, model):
    labels = labels.to('cuda')
    images = images.to('cuda')
    predictions = model(images, labels)
    loss = model.loss_function(predictions, labels)
    return predictions.detach(), loss.detach()


def main(config):
    device = torch.device('cuda')
    writer = SummaryWriter(config.logdir + 'K{}_N{}_B{}_S{}_L{}_O{}_R{}_M{}_D{}'.format(config.num_samples, config.num_classes, config.meta_batch_size, config.training_steps, config.learing_rate, config.optimizer, config.learing_rate, config.model_size, config.dnc))

    # Download Omniglot Dataset
    if not os.path.isdir('./omniglot_resized'):
        gdd.download_file_from_google_drive(file_id='1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI',
                                            dest_path='./omniglot_resized.zip',
                                            unzip=True)
    assert os.path.isdir('./omniglot_resized')

    # Create Data Generator
    data_generator = DataGenerator(config.num_classes, 
                                   config.num_samples, 
                                   device=device)

    # Create model and optimizer
    model = MANN(config.num_classes, config.num_samples, 
                 model_size=config.model_size, use_dnc=config.dnc)
    model.to(torch.double).to(device)
    if config.dnc == True:
    	model.to(torch.float).to(device)

    l_rate = config.learing_rate
    if config.optimizer == 'SGD':
        optim = torch.optim.SGD(model.parameters(), lr = l_rate)
    elif config.optimizer == 'RMSPROP':
        optim = torch.optim.RMSPROP(model.parameters(), lr = l_rate)
    else:
        optim = torch.optim.Adam(model.parameters(), lr = l_rate)
    
    for step in range(config.training_steps):
        images, labels = data_generator.sample_batch('train', config.meta_batch_size)
        images = images.to('cuda')
        labels = labels.to('cuda')
        _, train_loss = train_step(images, labels, model, optim)

        if (step + 1) % config.log_every == 0:
            images, labels = data_generator.sample_batch('test', 
                                                         config.meta_batch_size)
            images = images.to('cuda')
            labels = labels.to('cuda')
            pred, test_loss = model_eval(images, labels, model)
            pred = torch.reshape(pred, [-1, 
                                        config.num_samples + 1, 
                                        config.num_classes, 
                                        config.num_classes])
            pred = torch.argmax(pred[:, -1, :, :], axis=2)
            labels = torch.argmax(labels[:, -1, :, :], axis=2)
            
            writer.add_scalar('Train Loss', train_loss.cpu().numpy(), step)
            writer.add_scalar('Test Loss', test_loss.cpu().numpy(), step)
            writer.add_scalar('Meta-Test Accuracy', 
                              pred.eq(labels).double().mean().item(),
                              step)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--meta_batch_size', type=int, default=128)
    parser.add_argument('--training_steps', type=int, default=10000)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--model_size', type=int, default=128)
    parser.add_argument('--layer_type', type=str, default='LSTM')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--learing_rate', type=float, default=1e-3)
    parser.add_argument('--dnc', type=bool, default=False)
    parser.add_argument('--logdir', type=str, 
                        default='run/log/')
    main(parser.parse_args())