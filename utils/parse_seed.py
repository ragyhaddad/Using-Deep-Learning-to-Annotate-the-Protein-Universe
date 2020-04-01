#!/bin/bash/env python 
import sys,os,json 
import glob 

# Pfam -> Int
label_dict = {} 
with open('labels.txt',"r") as f:
    for line in f:
        line = line.strip() 
        cols = line.split('\t') 
        code,fam = cols[0],cols[1] 
        if fam not in label_dict:
            label_dict[fam] = int(code)

in_dir = [f for f in glob.glob(sys.argv[1]+'/*')]  
print(label_dict)
# Encode Family to Int Label 
def encode_label(inlabel):
    label = inlabel  
    p_index = label.index('.') 
    label = label[0:p_index] 
    try:
        label = label_dict[label] 
    except KeyError:
        print(label)
        label = 0
    
    return label
# Parse Seed File from Paper 
def parse_seeds(input_path,generator=False):
    infile = input_path 
    seqs = [] 
    labels = [] 
    if '.tsv' in input_path:
        return 
    outpath = input_path + '.tsv' 
    outfile = open(outpath,"w")
    with open(infile,"r") as f: 
        for i,line in enumerate(f): 
            if i == 0:
                continue
            line = line.strip() 
            cols = line.split(',') 
            family = cols[2] 
            family = int(encode_label(family))
            seq = cols[4]   
            outfile.write('%i\t%s\n' % (family,seq))  
    outfile.close() 

def convert_dir(in_dir):
    for file in in_dir: 
        parse_seeds(file)  

convert_dir(in_dir)
print('-- Parsed Seed to TSV')