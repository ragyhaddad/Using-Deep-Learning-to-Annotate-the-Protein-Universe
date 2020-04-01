#!/bin/bash/env python 
import torch 
from torch import nn 
from torch import optim
import sys,os,json 
import pickle 
import torch.utils.data as data_utils  
import numpy as np 
import torch.nn.functional as F 
import torchvision.models as models 
from utils import seqParser
import glob 

class DataLoader():
    def __init__(self,train_directory):
        self.trainDirectory = train_directory 
        self.train_files = [f for f in glob.glob(self.trainDirectory + '/*.tsv')]
        self.current_index = 0
        self.dataset_range = len(self.train_files) 
        self.current_dataset = self.train_files[self.current_index] 
    def next_dataset(self):
        if self.current_index == self.dataset_range - 1 : 
            return False,self.dataset_range 
        self.current_dataset = self.train_files[self.current_index]
        ds = seqParser.Dataset(fpath=self.current_dataset,seqlen=2094)
        num_seqs = ds.NSEQS
        x_train,y_train = ds.parse_data(batch_size=ds.NSEQS) # Returns a dataset (seq,label)
        y_train = torch.from_numpy(y_train)
        x_train = torch.from_numpy(x_train)
        x_train = torch.reshape(x_train,(x_train.shape[0],1,21,2094)) # Where 21 is number of rows 
        train = data_utils.TensorDataset(x_train,y_train)
        self.current_index += 1 # Itterate till next dataset 
        return ( train,num_seqs )


     

