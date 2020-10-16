#!/usr/bin/env python
# coding: utf-8

#IMPORTS
from __future__ import print_function
#Basics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy import stats
import locale
locale.setlocale(locale.LC_ALL, '')
import datetime
import io

#Biopython
from Bio.Align import MultipleSeqAlignment
from Bio import SeqIO
from Bio.pairwise2 import format_alignment
from Bio import pairwise2
from Bio import Entrez
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, FeatureLocation

#my modules
from modules import Annotation_project_functions as func
from modules import Datasets as mydatasets
from modules import NN_models as mynn
from enum import IntEnum
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import multilabel_confusion_matrix
#pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential
from torch.utils.data import BatchSampler, SequentialSampler
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import argparse
# tensorboard
from torch.utils.tensorboard import SummaryWriter


# argument parser
parser = argparse.ArgumentParser(description="NN run config")

# path to kmers df input
parser.add_argument("-i", "--kmerInput",  required=False, type=str, default = '/Users/lou/Data/Output/kmers_generator/example_labelled_kmer_df.csv', help = "path for kmers dataframe, default example")

# input dimension
parser.add_argument("-id", "--dimInput",  required=False, type=int, default = 100, help = "dimension of input embedding, default 100")

# input dimension
parser.add_argument("-od", "--dimOutput",  required=False, type=int, default = 4, help = "dimension of output target, default 2")

# Hidden layer dimension range
parser.add_argument("-d", "--dimLayer",  required=False, type=int, default = 128, help = "dimension of hidden layer(s), default 128")

# number of hidden layers range
parser.add_argument("-n", "--numLayer",  required=False, type=int, default = 4, help = "number of hidden layers, default 4")

# number of epochs
parser.add_argument("-e", "--numEpochs",  required=False, type=int, default = 50, help = "number of epochs, default 50")

# learning rate range
parser.add_argument("-l", "--lr",  required=False, type=float, default = 0.00001, help = "learning rate, default 1^e-5")

# batch size range
parser.add_argument("-b", "--batchSize",  required=False, type=int, default =20, help = "smallest batch size, default 20")

# number of kmers used to run the nn
parser.add_argument("-k", "--numKmers", required=False, type=int, default =10000, help = "number of kmers used to run the nn")

# path to tensorboard events directory
parser.add_argument("-t", "--tensorboard",  required=False, type=str, default = './tensorboard', help = "path for tensorboard events dir, default ./tensorboard ")

# size of subset data
parser.add_argument("-s", "--subsetSize",  required=False, type=int, default =0, help = "size of the data subset used for the run. Default = 0 means entire dataset")

# Loss weights
parser.add_argument("-w", "--weights",  required=False, type=str, default ='1 1', help = "Give 2 floats for loss weights. First 1 for class non genic, second one for genic")

# Random seed
parser.add_argument("-r", "--randomSeed",  required=False, type=int, default = 1, help = "Random seed number. Default = 1")


# commit parser
args = parser.parse_args()



#Get loss weights
w = args.weights
w = w.split()
w = [float(i) for i in w]


# In[2]:


# Define where to run the progeam (GPU if available, otherwise CPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# if subsetsize == 0 load entire data
if (args.subsetSize == 0):
    print('Loading kmers ...')
    kmers_df = pd.read_csv(args.kmerInput)
#if value given load subset
elif(args.subsetSize > 0):
    print('Loading subset of kmers ...')
    kmers_df = pd.read_csv(args.kmerInput, nrows = args.subsetSize)
else:
    print('subset size wrong value')

print('...')
kmers_df.drop(['Unnamed: 0'], axis = 1, inplace = True)
#convert embeddings to numerics
kmers_df.Embedding  = kmers_df.Embedding.apply(lambda y : pd.to_numeric(y.strip('[]').split()))
print('kmers loaded')

# set model
model = mynn.LSTMNet(args.dimInput, args.dimLayer, args.numLayer, args.dimOutput)
# send it to CPU or GPU
model.to(device)

#Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr= args.lr)

# set reporting files
path_to_log_dir = args.tensorboard
writer = SummaryWriter(log_dir = path_to_log_dir, filename_suffix= "binary_loss")
record = 'batch-'+str(args.batchSize)+'/lr-'+str(args.lr)+'/dimL-'+str(args.dimLayer)+'/nbL-'+str(args.numLayer)+'/nbKmers-'+str(args.numKmers)+'/Weights-'+str(w)

# running model
tr_loss, te_loss, pred, true = mynn.run_network_binary(kmers_df, model, args.numEpochs, optimizer, args.batchSize, loss_weights = w, nb_kmers = args.numKmers, tensorboard_log_dir = args.tensorboard, event_name = record, device = device, rdseed = args.randomSeed)
