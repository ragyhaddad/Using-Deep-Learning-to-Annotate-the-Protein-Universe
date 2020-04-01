#!/bin/bash/env python
# Author: Ragy Haddad
import torch 
from torch import nn 
from torch import optim
import sys,os,json,argparse
import pickle 
import torch.utils.data as data_utils  
import numpy as np 
import torch.nn.functional as F 
import torchvision.models as models 
from utils import seqParser 
from Bio import SeqIO 
import numpy as np 

# Globals
model = None
labels_dict = {}

# Run on GPU
device = torch.device('cuda:0')
def load_model(path):
    print('-- Loading Model')
    # Resnet with minor adjustments 
    checkpoint = torch.load(path) 
    model = models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=8, stride=2, padding=3,bias=False)
    model.fc = nn.Linear(in_features=512, out_features=17931, bias=True)
    model.load_state_dict(checkpoint['model_state_dict']) 
    loss = checkpoint['loss'] 
    epoch = checkpoint['epoch']

    model.to(device) 
    model.eval()
    return model 



def parse_labels():
    with open('utils/labels.txt',"r") as f:
        for line in f:
            line = line.strip()
            code,pfam = line.split('\t')[0:2]
            labels_dict[int(code)] = pfam 

# Data
num_classes = 17931 # Number of Families in Pfam Release 32.0 

def predict(seq): 
    global model
    arr = np.zeros((21,2094))
    #seq = 'VAGSLIGLFGLFGNASTALILTRPAMRNPNNLFLTALAVFDSCLLITAFFIYAMEYIIEYTRAFDLYVAWLTYLRFAFALSHISQTGSVYITVSVTIERYLAVCHPRRSKQMCNPGGAAWTILGVTTFAVLFNATKFFELEVTVNPACPEGKDWQSYILLPSVMAANPIYQQVYALWLTNIVMVFLPFLTLLILNAYIAYTIRRSLKKFDNHQQKLPDRSELKEKSREATLVLVIIVCIFLICNFWGFVLTLLERIVDHETLMVKYHAFYTFSREAINFLAIINSSINFVIYIVFGREFRKELVIVYGCG'
    x = seqParser.encode_seq(arr,seq) 
    x = np.array(x) 
    x = torch.from_numpy(x)
    x = x.reshape(1,1,21,2094) 
    x = x.to(device,dtype=torch.float)
    output = model(x)
    _, predicted = torch.max(output.data, 1)
    preds = torch.topk(output.data,10) 
    print('-- Best hit: ')
    print(labels_dict[predicted.item()]) 
    print('-- Integer Code: ')
    print(predicted.item())
    print('-- Top Predictions: ')
    for x in preds[1][0]:
        pred = labels_dict[x.item()]
        print(pred)

def main():
    global model
    argument_parser = argparse.ArgumentParser(description='Use DBTX Resnet to annotate Pfam Domains')  
    argument_parser.add_argument('-i','--input',help='Input String or Fasta File Path',required=True)
    argument_parser.add_argument('-mpath','--model_path',help='Mode: string or fasta' ,required=True)
    argument_parser.add_argument('-m','--mode',help='Mode: string or fasta' ,required=True)  
    args = vars(argument_parser.parse_args())
    mode = args['mode']
    model = load_model(args['model_path'])
    parse_labels()
    if mode == 'string':
        input_seq = args['input'].upper() 
        predict(input_seq)
    if mode == 'fasta':
        input_path = args['input'] 
        for rec in  SeqIO.parse(input_path,'fasta'):
            input_seq = rec.seq
            header = rec.description
            print(header)
            predict(input_seq) 
        

if __name__ == '__main__':
    main()