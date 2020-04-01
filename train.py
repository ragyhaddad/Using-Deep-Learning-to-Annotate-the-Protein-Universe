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
from utils import dataLoader

# Data
num_classes = 17931 # Number of Classes in Pfam 32.0 for softmax
DL = dataLoader.DataLoader(sys.argv[1]) # Takes in the directory of the split training files 
train,num_seqs = DL.next_dataset()
mini_batch = 100
train_loader = data_utils.DataLoader(train,batch_size=mini_batch,shuffle=True) 


# Resnet with minor adjustments
model = models.resnet18()
model.conv1 = nn.Conv2d(1, 64, kernel_size=8, stride=2, padding=3,bias=False)
model.fc = nn.Linear(in_features=512, out_features=17931, bias=True)  

# Select GPU device
device = torch.device('cuda:0')
net = model
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=0.001) # L2 Penalty 0.0005 

def load_model():
    global optimizer
    # Resnet with minor adjustments 
    checkpoint = torch.load('resnet_pfam_final.mdl') 
    model = models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=8, stride=2, padding=3,bias=False)
    model.fc = nn.Linear(in_features=512, out_features=17931, bias=True)
    model.load_state_dict(checkpoint['model_state_dict']) 
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss'] 
    epoch = checkpoint['epoch']
    model.train()
    model.to(device) 
    return model
# Train Method 
glob_loss = 9999 # Checkpoint Loss 
glob_epoch = 0
def train_net(net,epochs=50,load=False):
    global train_loader,num_seqs
    sys.stdout.write('-- Starting Training\n')
    temp_loss = 1.0
    correct = 0 
    epoch = 0
    if load == True:    
        print('-- Loading Model')
        net = load_model()
    while epoch < (epochs):
        running_loss = 0.0 
        for i,data in enumerate(train_loader,0):
            inputs,labels = data 
            # Move Tensors to GOU 
            inputs,labels = inputs.to(device,dtype=torch.float), labels.to(device,dtype=torch.float)  
            optimizer.zero_grad() 
            outputs = net(inputs) 
            labels = labels.long()
            loss = criterion(outputs,labels) 
            _, predicted = torch.max(outputs.data, 1) 
            correct += (predicted == labels).sum().item()
            loss.backward()
            optimizer.step() 
            running_loss += loss.item()  
            if i  % mini_batch == 0 :
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / mini_batch)) 
                glob_loss = running_loss
                running_loss = 0.0 
        acc = 100 * float(correct/num_seqs)
        print('Epoch: %i | Accuracy: %.6f' % (epoch,acc))
        correct = 0
        if epoch == epochs - 1:
            print('-- Switching Datasets')
            epoch = 0 
            train_switch,num_seqs = DL.next_dataset()
            print('-- Training on %s' % (DL.current_dataset))
            if train_switch == False:
                print('-- Training complete on all Datasets')
                return
            else:
                train_loader = data_utils.DataLoader(train_switch,batch_size=mini_batch,shuffle=True)
        # epoch += 1


# Train Model  
main_epochs = 6
for x in range(main_epochs):
    print('Main Epoch %i/%i' % (x+1,main_epochs))
    train_net(net,epochs=1,load=False)
    DL.current_index = 0
    

print('-- Saving Final Model')
# Save Model 
torch.save({
            'epoch': 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': glob_loss,
            }, 'resnet_pfam_final_8.mdl')
 

