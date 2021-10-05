import argparse
import os
import torch

import torch.nn.functional as F

from torch import nn
from load_data import DataGenerator
from dnc import DNC
from google_drive_downloader import GoogleDriveDownloader as gdd
from torch.utils.tensorboard import SummaryWriter


class MANN(nn.Module):

    def __init__(self, num_classes, samples_per_class, model_size=128, input_size=784):
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
                       )

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
        
        # reshape tensors to process
        reshaped_image = input_images.reshape(B * N * (K+1), dim)
        reshaped_label = input_labels.reshape(B * N * (K+1), N)
        print(reshaped_image.shape) # (88, 784)
        print(reshaped_label.shape) # (88, 4)
        
        # conatenate query image with label of zeros
        # Notes: in the previous part, we put all the query set of each character in each batch to the bottom, for each batch we have (1 * N) at the
        # bottom of input [B, K + 1, N, 784] are the query set
        # E.g: n = 4 classes, k = 10 ways, batch = 2, we have output shape of image batch is (2, 11, 4, 784)
        # For first batch, if we resize the tensor to (1, 44, 784), the last three vectors of index 41, 42, 43, 44 are in the query set
        # And if we resize tensor with both batch to (2 * 44, 784) = (88, 784) the query set is in position (41, 42, 43, 44) and (85, 86, 87, 88)
        # That are the indexes that we need to concatenate with vector of zeros, we can start to change the labels of these indexes before concatenating
        for i in range(len(reshaped_label)):
            # find last set of batch label, for e.g. example above, index should be 40, 41, 42, 43 and 84, 85, 86, 87
            if (i + 1) % (N * (K + 1)) == 0:
                for j in range(i, i - 4, -1):
                    reshaped_label[j] = np.zeros(N)

        
        # concatenate image and label to a tensor of [B * (K+1) * N, dim + N]
        concatenated = np.concatenate((reshaped_image, reshaped_label), axis = 1)
        
        # reshape concatenated tensor to [B, K+1, N, dim + N]
        concatenated = np.reshape(concatenated, (B, (K+1) * N, dim + N))
        concatenated = torch.from_numpy(concatenated).to(torch.double)
        
        # use print to confirm that query label have values of zeros
        
        #print(concatenated[0][40][783:]) # label should be zero 
        #print(concatenated[0][41][783:]) 
        #print(concatenated[0][42][783:]) 
        #print(concatenated[0][43][783:]) 
        
        # pass data through model and reshape output the [B, K + 1, N, N]
        output, _ = self.layer1(concatenated)
        output, _ = self.layer2(output)
        output = torch.reshape(output, (B, K + 1, N, N))
        
        return output


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
        preds_N = reshaped_preds.detach().numpy()[(len(reshaped_preds) -  B*N):]
        labels_N = reshaped_labels.detach().numpy()[(len(reshaped_preds) -  B*N):]
        
        # 8 * 4 # one-hot encoding for the label [B*N, N] and [B*N,]
        # transfor to 8 * 1
        labels_N = torch.argmax(torch.from_numpy(labels_N), dim = 1)
        preds_N = torch.from_numpy(preds_N)
        
        output = F.cross_entropy(preds_N, labels_N)

        return output    
        



def train_step(images, labels, model, optim):
    predictions = model(images, labels)
    loss = model.loss_function(predictions, labels)
    
    optim.zero_grad()
    loss.backward()
    optim.step()
    return predictions.detach(), loss.detach()


def model_eval(images, labels, model):
    predictions = model(images, labels)
    loss = model.loss_function(predictions, labels)
    return predictions.detach(), loss.detach()


def main(config):
    device = torch.device("cuda")
    writer = SummaryWriter(config.logdir)

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
                 model_size=config.model_size)
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr = 1e-3)
    
    for step in range(config.training_steps):
        images, labels = data_generator.sample_batch('train', config.meta_batch_size)
        _, train_loss = train_step(images, labels, model, optim)

        if (step + 1) % config.log_every == 0:
            images, labels = data_generator.sample_batch('test', 
                                                         config.meta_batch_size)
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
    parser.add_argument('--logdir', type=str, 
                        default='run/log')
    parser.add_argument('--training_steps', type=int, default=10000)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--model_size', type=int, default=128)
    main(parser.parse_args())