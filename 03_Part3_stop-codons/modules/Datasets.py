
####################################################
### IMPORTS
####################################################

#basics

import torch
from itertools import product
import os.path as osp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy import stats
import locale
import datetime
import argparse
import io

#Biopython
from Bio.Align import MultipleSeqAlignment
from Bio import SeqIO
from Bio.pairwise2 import format_alignment
from Bio import pairwise2
from Bio import Entrez
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, FeatureLocation

#Deep learning
from enum import IntEnum
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

####################################################
### DATASETS
####################################################



# Defines a dataset class for binary kmers dataframe in general
class KmersDataset(Dataset):

    def __init__(self,embedding_col,label_col, size = 10000):

        lenEmbed = len(embedding_col[0])
        v = []
        # generate a random kmer nb to pick random chunck of data in whole dataset
        rand_kmers= np.random.randint(0, high=len(label_col)-size)

        for embed in embedding_col[rand_kmers:rand_kmers+size] :
            for item in embed:
                v.append(item)

        X = np.array(v)
        Y = np.array(label_col[rand_kmers:rand_kmers+size].astype(int))

        dtype = torch.FloatTensor

        # X and Y into Variables, requires_grad false cause non learnable param
        # self[0]
        self.samples = Variable(torch.from_numpy(X).type(dtype), requires_grad = False).view(-1,lenEmbed)
        # self[1]
        self.labels = Variable(torch.from_numpy(Y).type(dtype), requires_grad = False).view(len(self.samples),1)

        # Define binary weights
        # nb_classes = 2

        prop0 = len(self.labels[self.labels == 0])/len(self.labels)
        prop1 = len(self.labels[self.labels == 1])/len(self.labels)

        prop_dict =   {0:prop0,1:prop1}

        props = Variable(torch.from_numpy(Y).type(dtype), requires_grad = False).view(len(self.samples),1)
        for i, label in enumerate(props):
            props[i] = prop_dict[int(label)]

        # self[2]
        self.prop = props



    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx], self.prop[idx]


#######
# Defines a dataset class for multiclass kmers dataframe (genic forward, genic reverse, genic both or non genic)
class MultiClassKmersDataset(Dataset):

    def __init__(self,df,size = 10000):

        lenEmbed = len(df.Embedding[0])
        v = []
        # generate a random kmer nb to pick random chunck of data in whole dataset
        rand_kmers= np.random.randint(0, high=len(df)-size)

        for embed in df.Embedding[rand_kmers:rand_kmers+size] :
            for item in embed:
                v.append(item)

        embeddings = np.array(v)
        genic_f = np.array(df.Gene_dir_forward[rand_kmers:rand_kmers+size].astype(int))
        genic_r = np.array(df.Gene_dir_reverse[rand_kmers:rand_kmers+size].astype(int))

        dtype = torch.FloatTensor

        # Embeddings and  into Variables, requires_grad false cause non learnable param
        # self[0]
        self.samples = Variable(torch.from_numpy(embeddings).type(dtype), requires_grad = False).view(-1,lenEmbed)
        # self[1]
        self.forward = Variable(torch.from_numpy(genic_f).type(dtype), requires_grad = False).view(len(self.samples),1)
        # self[2]
        self.reverse = Variable(torch.from_numpy(genic_r).type(dtype), requires_grad = False).view(len(self.samples),1)

        #self[3]
        kmer_classe = []
        for i in range(len(self.samples)):
            f = int(self.forward[i].numpy())
            r = int(self.reverse[i].numpy())
            if ((f == 1) and (r == 0)):
                kmer_classe.append(1)
            elif ((f == 0) and (r == 1)):
                kmer_classe.append(2)
            elif ((f == 1) and (r == 1)):
                kmer_classe.append(3)
            elif ((f == 0) and (r == 0)):
                kmer_classe.append(0)

        self.kmerClass = Variable(torch.tensor(kmer_classe).type(dtype), requires_grad = False).view(len(self.samples),1)

        # Define class weights
        prop0 = len(self.kmerClass == 0)/len(self.kmerClass)
        prop1 = len(self.kmerClass == 1)/len(self.kmerClass)
        prop2 = len(self.kmerClass == 2)/len(self.kmerClass)
        prop3 = len(self.kmerClass == 3)/len(self.kmerClass)
        max_prop = max(prop0,prop1,prop2,prop3)
        prop_list = [prop0,prop1,prop2,prop3]

        # self [4] : proportions of each category
        self.prop = [prop0,prop1,prop2,prop3]

        # avoid getting null division for proposed prop
        for i, prop in enumerate(prop_list) :
            if prop == 0:
                print('prop',i,' is null')
                # really high number to get a weight close to zero ( which does not change anything)
                prop_list[i] = 10000

        weight_dict = {'nonGenic':max_prop/prop_list[0],'forward':max_prop/prop_list[1],'reverse':max_prop/prop_list[2],'bothStrands':max_prop/prop_list[3]}

        # self[5]: proposed loss weights
        self.weights = weight_dict

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.forward[idx], self.reverse[idx], self.kmerClass[idx]



######
# Defines a dataset class for binary kmers dataframe with non random kmer  size 
class predDataset(Dataset):

    def __init__(self,embedding_col):

        lenEmbed = len(embedding_col[0])

        X = np.array(embedding_col)

        dtype = torch.FloatTensor

        # X and Y into Variables, requires_grad false cause non learnable param
        # self[0]
        self.samples = Variable(torch.from_numpy(X).type(dtype), requires_grad = False).view(-1,lenEmbed)


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        return self.samples[idx]
