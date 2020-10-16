#!/usr/bin/env python
# coding: utf-8

# script running LSTM model on stop k-mers

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

import datetime
import io
import random

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
parser.add_argument("-i", "--kmerInput",  required=False, type=str, default = '/tudelft.net/staff-umbrella/abeellab/lou/Annotation_Project/02_Part2_Start-codon/01_Generate_kmers/output/StartCodons_reads_+-10_10mers_20200507-17-27-31_shuffled', help = "path for kmers dataframe")

# input dimension
parser.add_argument("-id", "--dimInput",  required=False, type=int, default = 100, help = "dimension of input embedding, default 100")

# input dimension
parser.add_argument("-od", "--dimOutput",  required=False, type=int, default = 1, help = "dimension of output target, default 1")

# Hidden layer dimension range
parser.add_argument("-d", "--dimLayer",  required=False, type=int, default = 128, help = "dimension of hidden layer(s), default 128")

# number of hidden layers range
parser.add_argument("-n", "--numLayer",  required=False, type=int, default = 4, help = "number of hidden layers, default 4")

# number of epochs
parser.add_argument("-e", "--numEpochs",  required=False, type=int, default = 20, help = "number of epochs, default 20")

# learning rate range
parser.add_argument("-l", "--lr",  required=False, type=float, default = 0.00001, help = "learning rate, default 1^e-5")

# batch size range
parser.add_argument("-b", "--batchSize",  required=False, type=int, default =20, help = "smallest batch size, default 20")

# number of kmers used to run the nn
parser.add_argument("-k", "--numKmers", required=False, type=int, default =60000, help = "number of kmers used to run the nn")

# path to tensorboard events directory
parser.add_argument("-t", "--tensorboard",  required=False, type=str, default = './tensorboard', help = "path for tensorboard events dir, default ./tensorboard ")

# size of subset data
parser.add_argument("-s", "--subsetSize",  required=False, type=int, default =0, help = "size of the data subset used for the run. Default = 0 means entire dataset")

# Loss weights
parser.add_argument("-w", "--weights",  required=False, type=str, default ='1 1', help = "Give 2 floats for loss weights. First 1 for class non genic, second one for genic")

# Random seed
parser.add_argument("-r", "--randomSeed",  required=False, type=int, default = 1, help = "Random seed number. Default = 1")

# Path for saving model
parser.add_argument("-om", "--outpathModel",  required=False, type=str, default = './', help = "Path to save model. Default = './' ")

# Model to load
parser.add_argument("-m", "--model",  required=False, type=str, default = 'None', help = "If a model is given then the model is loaded from the path provided. Default = None")


# commit parser
args = parser.parse_args()

#Get loss weights
w = args.weights
w = w.split()
w = [float(i) for i in w]

# model path
model_path = args.outpathModel

# load kmers :
kmers_file = args.kmerInput
kmers_df = pd.read_csv(kmers_file)
kmers_df.drop(['Unnamed: 0'], axis = 1, inplace = True)

#convert embeddings to numerics
kmers_df.drop(kmers_df[kmers_df['Embedding'] == "None"].index, axis = 0, inplace = True)
kmers_df.Embedding  = kmers_df.Embedding.apply(lambda y : pd.to_numeric(y.strip('[]').split()))

# Define where to run the progeam (GPU if available, otherwise CPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# init parameters
dim_input = args.dimInput
dim_recurrent= args.dimLayer
num_layers = args.numLayer
dim_output = args.dimOutput
learning_rate = args.lr
epochs = args.numEpochs
batch_size = args.batchSize
loss_weights = w
seed = args.randomSeed
nb_kmers = args.numKmers

model = mynn.LSTMNet(dim_input, dim_recurrent, num_layers, dim_output)
# load model if needed
if args.model != 'None':
    print('Model loading :', args.model)
    model.load_state_dict(torch.load(args.model))
# send it to CPU or GPU
model.to(device)

# set reporting record
record = 'CodonStarts-reads_dimRec-'+str(dim_recurrent)+'_numLayer-'+str(num_layers)+'_lr-'+str(learning_rate)+'_Weights-'+str(loss_weights)+'_batchSize-'+str(batch_size)+'_rdSeed-'+str(seed)

#set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#run model
tr_loss, ts_loss, pred, true, updated_model = mynn.run_network_binary(kmers_df.Embedding, kmers_df.Is_stop_codon, model, epochs, optimizer, batch_size,loss_weights = loss_weights, nb_kmers = nb_kmers,tensorboard_log_dir = args.tensorboard, event_name =record, device = device, rdseed = seed)

# save model
torch.save(updated_model.state_dict(), model_path)
