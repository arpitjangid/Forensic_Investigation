import os
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
# from scipy.io import loadmat 
import csv
import torch

class TripletFID(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, fid_dataset):
        self.fid_dataset = fid_dataset
        self.train = self.fid_dataset.train
        # self.transform = self.fid_dataset.transform
        
        self.transform_ref = self.fid_dataset.transform_ref
        self.transform_probe = self.fid_dataset.transform_probe
        
        self.num_reference = self.fid_dataset.num_reference
        self.reference_filenames = self.fid_dataset.reference_filenames
        self.labels = self.fid_dataset.labels
        self.probe_images = self.fid_dataset.probe_images
        self.reference_images = self.fid_dataset.reference_images
        
        if self.train:
            random_state = np.random.RandomState()
        else:
            random_state = np.random.RandomState(29)
        # probe, positive, negative
            
        triplets = []
        for i in range(len(self.probe_images)):
            positive_ind = self.labels[i].item()
            # negative_ind = i
            # while(negative_ind != i):
            negative_ind = self.labels[i].item() #i #  initializing with pos_ind
            while(negative_ind == positive_ind):
                negative_ind = random_state.randint(self.num_reference)
            triplet = [i, positive_ind, negative_ind]
            triplets.append(triplet)
        self.triplets = triplets
        

    def __getitem__(self, index):
        img1 = self.probe_images[self.triplets[index][0]] # anchor
        img2 = self.reference_images[self.triplets[index][1]] # positive
        img3 = self.reference_images[self.triplets[index][2]] #negative

        img1 = Image.fromarray(np.array(img1), mode='RGB')
        img2 = Image.fromarray(np.array(img2), mode='RGB')
        img3 = Image.fromarray(np.array(img3), mode='RGB')
        if self.transform_probe is not None:
            img1 = self.transform_probe(img1)
        if self.transform_ref is not None:
            img2 = self.transform_ref(img2)
            img3 = self.transform_ref(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.fid_dataset)


class FID300(Dataset):
    """
    A customized data loader for FID300
    """
    def __init__(self,
                 root,
                 transform=None,
                 train = True, get_probe = True, with_aug = False):#,preload=False):
        """ Intialize the FID300 dataset
        
        Args:
            - root: root directory of the dataset
            - tranform: a custom tranform function
            - preload: if preload the dataset into memory
        """
        self.probe_images = None
        self.reference_images = None
        self.labels = None
        self.reference_filenames = []
        self.probe_filenames = []
        self.root = root
        # self.transform = transform
        self.train = train # train set or test set
        if self.train:
            self.transform_ref = transform[0]
            self.transform_probe = transform[1]
        else:
            self.transform_ref = transform
            self.transform_probe = transform
        
        self.get_probe = get_probe
        # self.get_ref = get_ref

        ref_dir = os.path.join(root,"references")
        ref_filenames = os.listdir(ref_dir)
        ref_filenames.sort()
        self.reference_filenames = [os.path.join(ref_dir,ref_file) for ref_file in ref_filenames]
        self.num_reference = len(ref_filenames)
        if(self.train):
            if(with_aug):
                probe_dir = os.path.join(root,"tracks_cropped_train_aug")
                label_file = os.path.join(root,'label_table_train_aug.csv')
            else:
                probe_dir = os.path.join(root,"tracks_cropped_train")
                label_file = os.path.join(root,'label_table_train.csv')
        else:
            probe_dir = os.path.join(root,"tracks_cropped_val")
            label_file = os.path.join(root,'label_table_val.csv')
        probe_flist = os.listdir(probe_dir)
        probe_flist.sort()
        probe_flist = [os.path.join(probe_dir, probe_file) for probe_file in probe_flist]

        label_map = np.loadtxt(label_file, delimiter=',', dtype='int')
        label_map[:,1] -= 1 # making the map to range [0,1174], 
        
        # probe_flist is in sorted order
        for i in range(len(probe_flist)):
            self.probe_filenames.append((probe_flist[i],label_map[i,1])) # filename and ref file_id pair
        # if preload dataset into memory
        #if preload:
        self._preload()
        if(self.get_probe):    
            self.len = len(self.probe_filenames)
        else:
            self.len = len(self.reference_filenames)
                              
    def _preload(self):
        """
        Preload dataset to memory
        """
        self.labels = []
        self.probe_images = []
        self.reference_images = []
        for image_fn, label in self.probe_filenames:            
            # load images
            # image = Image.open(image_fn)
            image = Image.open(image_fn).convert("RGB")
            self.probe_images.append(image.copy())
            # avoid too many opened files bug
            image.close()
            self.labels.append(label)
        for image_fn in self.reference_filenames:
        #   image = Image.open(image_fn)
            image = Image.open(image_fn).convert("RGB")
            self.reference_images.append(image.copy())
            image.close()
    

    # probably the most important to customize.
    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        # if self.probe_images is not None:
        # If dataset is preloaded, in this case data is preloaded
        if self.get_probe:
            image = self.probe_images[index]
            label = self.labels[index]
            # ref_image = self.reference_images[label-1]
            if self.transform_probe is not None:
                image = self.transform_probe(image)
            # return image and label
            return image, label
        else: # output reference images
            ref_image = self.reference_images[index]
            if self.transform_ref is not None:
                ref_image = self.transform_ref(ref_image)
            # return image and label
            return ref_image
        
            
        # else:
        #     # If on-demand data loading
        #     image_fn, label = self.probe_filenames[index]
        #     ref_image_fn = self.reference_filenames[label-1]
        #     image = Image.open(image_fn)
        #     ref_image = Image.open(ref_image_fn)
            
        # May use transform function to transform samples
        # e.g., random crop, whitening

        # if(~self.get_probe and ~self.get_ref):

        #     if self.transform is not None:
        #         image = self.transform(image)
        #         ref_image = self.transform(ref_image)
        #     # return image, ref_image and label
        #     return image, ref_image, label
        

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len
