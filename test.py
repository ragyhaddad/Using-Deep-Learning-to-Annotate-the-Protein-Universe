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

device = torch.device('cuda:0')
def load_model():
    # Resnet with minor adjustments 
    checkpoint = torch.load('resnet_pfam_final_8.mdl') 
    model = models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=8, stride=2, padding=3,bias=False)
    model.fc = nn.Linear(in_features=512, out_features=17931, bias=True)
    model.load_state_dict(checkpoint['model_state_dict']) 
    loss = checkpoint['loss'] 
    epoch = checkpoint['epoch']
    model.eval()
    model.to(device) 
    return model 

model = load_model()

# Data
num_classes = 17931 # Number of Families in Pfam Release 32.0 
ds = seqParser.Dataset(fpath=sys.argv[1],seqlen=2094)  
x_test,y_test = ds.parse_data(batch_size=ds.NSEQS) # Returns a dataset (seq,label) 
y_test = torch.from_numpy(y_test)
x_test = torch.from_numpy(x_test)
x_test = torch.reshape(x_test,(x_test.shape[0],1,21,2094)) # Where 21 is number of rows  
mini_batch = 4
test = data_utils.TensorDataset(x_test,y_test)
test_loader = data_utils.DataLoader(test,batch_size=mini_batch,shuffle=True) 


# Validation Method
def validate(net):
    correct = 0
    total = 0 
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes)) 
    with torch.no_grad(): # With torch No Gradient Descent
        for i,data in enumerate(test_loader):
            inputs, labels = data
            inputs,labels = inputs.to(device,dtype=torch.float), labels.to(device,dtype=torch.float)  
            labels = labels.long()
            # print(labels)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1) # Unpack Output of last layer and take index of the max weight 
            # print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze() 
            #print(i,labels[0].item(),predicted[0].item())  
    print('Total Samples: %i' % (total)) 
    print('Correct: %i' % (correct))    
    acc = 100 * float(correct/total) 
    print('-- Accuracy of the network on the test points: %.6f' % (acc)) 

validate(model)


