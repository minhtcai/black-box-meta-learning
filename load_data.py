import numpy as np
import os
import random
import torch


def get_images(paths, labels, nb_samples=None, shuffle=True):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    Args:
        paths: A list of character folders
        labels: List or numpy array of same length as paths
        nb_samples: Number of images to retrieve per character
    Returns:
        List of (label, image_path) tuples
    """
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images_labels = [(i, os.path.join(path, image))
                     for i, path in zip(labels, paths)
                     for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images_labels)
    return images_labels


def image_file_to_array(filename, dim_input):
    """
    Takes an image path and returns numpy array
    Args:
        filename: Image filename
        dim_input: Flattened shape of image
    Returns:
        1 channel image
    """
    import imageio
    image = imageio.imread(filename)
    image = image.reshape([dim_input])
    image = image.astype(np.float32) / 255.0
    image = 1.0 - image
    return image


class DataGenerator(object):
    """
    Data Generator capable of generating batches of Omniglot data.
    A "class" is considered a class of omniglot digits.
    """

    def __init__(self, num_classes, num_samples_per_class, config={}, device = torch.device('cuda')):
        """
        Args:
            num_classes: int
                Number of classes for classification (N-way)
            
            num_samples_per_class: int
                Number of samples per class in the support set (K-shot).
                Will generate additional sample for the querry set.
                
            device: cuda.device: 
                Device to allocate tensors to.
        """
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes

        data_folder = config.get('data_folder', './data/omniglot_resized')
        self.img_size = config.get('img_size', (28, 28))

        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes

        character_folders = [os.path.join(data_folder, family, character)
                             for family in os.listdir(data_folder)
                             if os.path.isdir(os.path.join(data_folder, family))
                             for character in os.listdir(os.path.join(data_folder, family))
                             if os.path.isdir(os.path.join(data_folder, family, character))]

        random.seed(1)
        random.shuffle(character_folders)
        num_val = 100
        num_train = 1100
        self.metatrain_character_folders = character_folders[: num_train]
        self.metaval_character_folders = character_folders[
            num_train:num_train + num_val]
        self.metatest_character_folders = character_folders[
            num_train + num_val:]
        self.device = device

    def sample_batch(self, batch_type, batch_size):
        """
        Samples a batch for training, validation, or testing
        Args:
            batch_type: str
                train/val/test set to sample from
                
            batch_size: int:
                Size of batch of tasks to sample
                
        Returns:
            images: tensor
                A tensor of images of size [B, K+1, N, 784]
                where B is batch size, K is number of samples per class, 
                N is number of classes
                
            labels: tensor
                A tensor of images of size [B, K+1, N, N] 
                where B is batch size, K is number of samples per class, 
                N is number of classes
        """
        if batch_type == "train":
            folders = self.metatrain_character_folders
        elif batch_type == "val":
            folders = self.metaval_character_folders
        else:
            folders = self.metatest_character_folders

        #############################
        #### YOUR CODE GOES HERE ####
        #############################
        
        # SOLUTION:
        # Sample N different character and labels from train, test, validation
        B = batch_size
        N = self.num_classes
        K = self.num_samples_per_class
        dim = self.dim_input 
        batch_images = []
        batch_labels = []
        
        # Pick number of task equal to batch
        for i in range(B):
            # Sample from folder with selected number of class
            sampled_class = random.sample(folders, N)
            
            # Load K+1 images per char and collect labels, using K images per class for support set and one image per class for the query class
            # Create label matrix of size N*N using identity matrix, since for each class will have it own correspondence label encoded
            labels_encoded = np.identity(N)
            #print(labels_encoded)
            
            # Collect image and labels with K+1 sample for each sampeld class, have shape
            labels_imgs = get_images(sampled_class, labels_encoded, K+1, shuffle=False) # N * (K_+ 1) 
            
            # Create tensor and load data in support and train
            #labels_imgs_matrix = np.reshape(labels_imgs, (K + 1, N, 784))
            
            support_set = [] # K * N * dim
            query_set = [] # 1 * N * dim
            support_set_label = [] 
            query_set_label = [] 
            
            # query will have shape 1 * N * dim
            # support will have shape K * N * dim
            # take first sample of each character batch for the query set
            test_counter = 0
            for j in range(len(labels_imgs)): 
                if j == test_counter:
                    query_set.append(image_file_to_array(labels_imgs[j][1], dim)) 
                    query_set_label.append(labels_imgs[j][0])
                    #print(labels_imgs[j][1])
                    #print(labels_imgs[j][0])
                    test_counter += (K+1)
                else:
                    support_set.append(image_file_to_array(labels_imgs[j][1], dim))
                    support_set_label.append(labels_imgs[j][0])
                    #print(labels_imgs[j][1])
                    #print(labels_imgs[j][0])
            
            
            # Shuffle query set only
            query_set, query_set_label = shuffle(query_set, query_set_label)
            #support_set, support_set_label = shuffle(support_set, support_set_label)
            
            # Put to images tensor (K + 1) * N * dim 
            images_matrix = np.concatenate((support_set, query_set), axis=0)
            #print(images_matrix.shape)
            images_matrix = images_matrix.reshape((K + 1, N, dim))
            #print(images_matrix.shape)
            
            # Put to labels tensor (K + 1) * N * N
            labels_matrix = np.concatenate((support_set_label, query_set_label), axis=0)
            #print(labels_matrix.shape)
            labels_matrix = labels_matrix.reshape((K + 1, N, N))
            #print(labels_matrix.shape)
            
            # Add to batch
            batch_images.append(images_matrix)
            batch_labels.append(labels_matrix)
            
        #print(np.asarray(batch_images).shape)
        #print(np.asarray(batch_labels).shape)
        
        # Format the data and return two matrices, one of flattened images with specified shape
        return batch_images, batch_labels