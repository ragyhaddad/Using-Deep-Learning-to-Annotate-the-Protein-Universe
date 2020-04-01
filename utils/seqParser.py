#!/bin/bash/env python 
import sys,os,json 
import pickle 
import numpy as np 
import torch 
import subprocess

# Row Wise Index for each AA 
CHARSET = { 'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, \
            'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, \
            'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20, \
            'O': 20, 'U': 20,
            'B': (2, 11),
            'Z': (3, 13),
            'J': (7, 9)}  
CHARLEN = 21 

def get_charset():
    return CHARSET 

def prepare_seq(seq):
    idxs = [CHARSET[c] for c in seq] 
    return torch.tensor(idxs,dtype=torch.long)
def seq_2_int(seq):
    seq = list(seq) 
    for i,char in enumerate(seq):
        seq[i] = CHARSET[char] 
"""
    ARGS:
        param1: (2D Numpy Array) CHARLEN * SEQLEN
        param2: (str) Sequence 
    RETURN:
        (2D Numpy Array) With One Hot Encoding
"""
def encode_seq(arr,seq):
    for pos,char in enumerate(seq):
        if char == '_': 
            continue 
        elif isinstance(CHARSET[ char ],int): 
            idx = CHARSET[ char ]
            arr[ idx ][ pos ] = 1  # Where idx is the row number in the matrix 
        else: 
            # Handle Ambiguous Amino Acids
            idx1 = CHARSET[ char ][0] 
            idx2 = CHARSET[ char ][1] 
            arr[ idx1 ][ pos ] = 0.5 
            arr[ idx2 ][ pos ] = 0.5 
    return arr 

def seq_to_categorical(seq,max_length,seq_np):
    for i,x in enumerate(list(seq)):
        seq_np[i] = int(CHARSET[x] + 1 ) 
    return (seq_np) 
""" 
    CLASS: Datset 
    Convert a TSV of The following Format to A Training Dataset 
    infile Format: Label\tSequence  

"""
class Dataset(object):
    def __init__(self,fpath,seqlen=2094,nseqs=100000,batch_size=70000):
        self.SEQLEN = seqlen
        self.NCLASSES = 0
        self.CHARSET = CHARSET 
        self.NSEQS = nseqs
        self.FILEPATH = fpath
        self.BATCHSIZE = batch_size 
        self._raw = self.read_raw(fpath) 
        self.NSEQS = self.count_seqs()
    # Count Number of Seqs In File 
    def count_seqs(self):
        output = subprocess.check_output("wc -l %s" % self.FILEPATH, shell=True).decode('utf-8')
        output = int(output.split(' ')[0])
        return output

    # Returns a Generator Object 
    def read_raw(self,fpath):
        x = []
        y = []
        with open(fpath,"r") as f:
            for i,line in enumerate(f.readlines()):
                line = line.strip()
                cols = line.split('\t')
                seq = cols[1]
                label = int(cols[0]) 
                x.append(seq) 
                y.append(label)
                yield(x,y) 
      
    # Parse to Matrix
    def parse_data(self,generator=False,batch_size=100): 
        data = np.zeros((batch_size,CHARLEN,self.SEQLEN), dtype=np.int32) # Empty Array with Size Fixed  
        labels = []
        matrix_index = 0  
        for i,x in enumerate(self.read_raw(self.FILEPATH)):  
            matrix_2d = data[i]
            seq = x[0][i]
            seq = seq.replace('_','')
            label = int(x[1][i]) 
            enc_matrix = encode_seq(matrix_2d,seq)
            data[matrix_index] = enc_matrix 
            labels.append(label) 
            matrix_index += 1  
            if i + 1 == batch_size:
                break
        labels = np.array(labels)
        if generator == False:
            return(data,labels)  
    
    # Parse to Categorical 
    def parse_data_linear(self,save=False):
        seqs = np.zeros( (self.NSEQS,self.SEQLEN) ,dtype=np.int32)
        labels = [] 
        for i,x in enumerate(self.read_raw(self.FILEPATH)):
            seq = x[0][i] 
            label = int(x[1][i]) 
            current_arr = seqs[i]
            seq = seq_to_categorical(seq,self.SEQLEN,current_arr) 
            seqs[i] = seq
            labels.append(label)
        return(seqs,labels)
     
    # Full Matrix Batch
    def full_batch(self,save=False): 
        x,y = self.parse_data() 
        x = np.array(x) 
        y = np.array(y)
        output = self.FILEPATH + '.pickle'
        with open(output, "wb") as f:
            pickle.dump([x,y],f)
            print('-- Pickle File Saved Successfully')
            print('-- Path: %s' % output)  

    # Full Catgorical Batch
    def full_batch_linear(self,save=False):
        x,y = self.parse_data_linear() 
        x = np.array(x)
        y = np.array(y)
        output = self.FILEPATH + '._cat.pickle' 
        with open(output,"wb") as f:
            pickle.dump([x,y],f) 
            print('-- Pickle File Saved Successfully')
            print('-- Path: %s' % output)




