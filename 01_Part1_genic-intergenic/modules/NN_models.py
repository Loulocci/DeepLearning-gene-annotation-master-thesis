####################################################
### IMPORTS
####################################################

#basics

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
import random

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import multilabel_confusion_matrix

#pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
#tensorboard
from torch.utils.tensorboard import SummaryWriter

#my modules
from modules import Annotation_project_functions as func
from modules import Datasets as mydatasets


####################################################
### NEURAL NETWORK CLASSES
####################################################

#########################
# class inspired by Francois Fleuret's teaching
class LSTMNet(nn.Module):

    # init function initialize layer and overall network structure
    # dim_recurrent = hidden dimension (size of hidden layers)
    def __init__(self, dim_input, dim_recurrent, num_layers, dim_output):

        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size = dim_input, hidden_size = dim_recurrent, num_layers = num_layers)
        # output layer
        self.out_layer = nn.Linear(dim_recurrent, dim_output)
        self.sg = nn.Sigmoid()

    # foward is the function that passes input through the network defined in init
    def forward(self, inp):

        # inp.shape : [seq_len,nb_features] ==> [seq_len,minibatch_size = 1, nb_features]
        # mini_batchsize = 1 because it s the number of kmers processed in parallele (here entire batch)
        #seq len = batch size = nb of words/kmers in my sequence ==> batch_first = False
        inp = inp.unsqueeze(1)
        # Get the activations of all layers at the last time step
        output, _ = self.lstm(inp)
        # Drop the batch index
        output = output.squeeze(1)
        #output = output[output.size(0) - 1:output.size(0)]
        output = self.out_layer(output)
        output = self.sg(output)
        return output


#################################
class multioutLSTMNet(nn.Module):
    # class defined for more than 2 classes prediction
    # init function initialize layer and overall network structure
    # dim_recurrent = hidden dimension (size of hidden layers)
    def __init__(self, dim_input, dim_recurrent, num_layers, dim_output):

        super(multioutLSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size = dim_input, hidden_size = dim_recurrent, num_layers = num_layers)
        # output layer
        self.out_layer = nn.Linear(dim_recurrent, dim_output)

    # foward is the function that passes input through the network defined in init
    def forward(self, inp):

        # Makes this a batch of size 1 : if 1*a becomes a*1
        inp = inp.unsqueeze(1)
        # Get the activations of all layers at the last time step
        output, _ = self.lstm(inp)
        # Drop the batch index
        output = output.squeeze(1)
        #output = output[output.size(0) - 1:output.size(0)]
        output = self.out_layer(output)
        output = F.softmax(output)

        return output

############################
# Class defined for bidirectional LSTMs
class BidLSTMNet(nn.Module):

    # init function initialize layer and overall network structure
    # dim_recurrent = hidden dimension (size of hidden layers)
    def __init__(self, dim_input, dim_recurrent, num_layers, dim_output):

        super(BidLSTMNet, self).__init__()
        self.dim_input = dim_input
        self.dim_recurrent = dim_recurrent
        self.num_layers = num_layers
        self.dim_output = dim_output

        self.bidirlstm = nn.LSTM(input_size = dim_input, hidden_size = dim_recurrent, num_layers = num_layers, bidirectional = True)
        # output layer
        self.out_layer = nn.Linear(2*dim_recurrent, dim_output)
        self.sg = nn.Sigmoid()
        self.softmax = nn.Softmax()

    # foward is the function that passes input through the network defined in init
    def forward(self, inp):

        # Makes this a batch of size 1 : if 1*a becomes a*1
        inp = inp.unsqueeze(1)
        # Get the activations of all layers at the last time step
        output, _ = self.bidirlstm(inp) # _ correspond to hidden states
        # Drop the batch index
        output = output.squeeze(1)
        #output = output[output.size(0) - 1:output.size(0)]
        output = self.out_layer(output)
        if self.dim_output > 1:
            output = self.softmax(output)
        else:
            output = self.sg(output)


        return output


####################################################
### RUNNING FUNCTIONS
####################################################

# basic running function
def run_network(data, model ,num_epochs, optim, batch_size = 20, nb_kmers = 10000, phase = 'Training', tensorboard_log_dir = './tensorboard', event_name = 'loss', device = torch.device('cpu')):

    # define tensorboard writer (reporting)
    now = datetime.datetime.now().strftime("%d-%m-%Y-%H%M%S")
    writer = SummaryWriter(log_dir = tensorboard_log_dir,filename_suffix= str(now))

    #load dataset
    data =  mydatasets.KmersDataset(data, size = nb_kmers) # build Kmers dataset object from data

    if phase == 'Training':
        model.train()
    if phase == 'Testing':
        model.eval()

    training_loss = []
    batch_step = 0
    for epoch in range(1, num_epochs+1):

        data_loader = DataLoader(data, batch_size= batch_size, shuffle = True)

        for batch_index, batch in enumerate(data_loader):

            #define variables to feed to model
            X = batch[0].to(device) # send to gpu
            labels = batch[1].to(device) # send to gpu
            loss_weight = batch[2].to(device)
            writer.add_scalar('Training/batch_mean/'+event_name,labels.mean(),batch_step)

            if phase == 'Training':
                optim.zero_grad() # reset the gradient

            out = model(X) # get the prediction from the model

            # Weighted loss function
            loss = F.binary_cross_entropy_with_logits(out,labels, weight = loss_weight)
            training_loss.append(loss)

            # Record training loss from each epoch into the writer
            writer.add_scalar('Training/loss/'+event_name, loss, batch_step)
            writer.flush()
            batch_step += 1

            if phase == 'Training':
                loss.backward()
                optim.step()

                # print loss progression
                if batch_index % 10 == 0:
                    print ('Epoch [{}/{}], Step [{}], Loss: {:.4f}'.format(epoch, num_epochs, batch_index, loss.item()))

    writer.close()
    return training_loss


###############################################################################
# upgraded running function with evaluation of the model while training
def run_network_binary(data, model, num_epochs, optim, batch_size, loss_weights = [1,1], nb_kmers = 10000, tensorboard_log_dir = './tensorboard',event_name = 'new_event', device = torch.device('cpu'), rdseed = 1):

    # prepare variables for reporting
    now = datetime.datetime.now().strftime("%d-%m-%Y-%H%M%S")
    writer = SummaryWriter(log_dir = tensorboard_log_dir,filename_suffix= str(now))

    #reproducibility
    torch.manual_seed(rdseed)
    np.random.seed(rdseed)

    #load dataset
    data =  mydatasets.KmersDataset(data, size = nb_kmers) # build Kmers dataset object from data
    train_set, test_set = train_test_split(data, test_size= 0.2)

    # Init storing arrays
    training_loss = []
    testing_loss = []
    pred = []
    true = []


    #define counter for reporting
    tr_batch_step = 0
    ts_batch_step = 0

    for epoch in range(1, num_epochs+1):

        model.train()
        data_loader = DataLoader(train_set, batch_size= batch_size, shuffle = True)

        for batch_index, batch in enumerate(data_loader):


            #define variables to feed the model
            samples = batch[0].to(device) # embedding sent to gpu
            labels = batch[1].to(device) # kmer class sent to gpu

            # prepare weights for loss function
            weights = []
            for item in labels :
                if int(item.data) == 0:
                    weights.append(loss_weights[0])
                if int(item.data) == 1:
                    weights.append(loss_weights[1])
            weights = torch.tensor(weights).unsqueeze(1)
            weights = weights.to(device)

            # run model
            optim.zero_grad()
            out = model(samples)
            #get loss
            loss = F.binary_cross_entropy_with_logits(out,labels, weight = weights)
            training_loss.append(loss)

            # Record training loss from each epoch into the writer
            writer.add_scalar('Training/Loss/'+event_name, loss, tr_batch_step)
            tr_batch_step += 1

            loss.backward()
            optim.step()

            if batch_index % 10 == 0:
                print ('Epoch [{}/{}], Step [{}], Training loss: {:.4f}'.format(epoch, num_epochs, batch_index, loss.item()))


    ########TESTING PHASE##########

        model.eval()

        data_loader_test = DataLoader(test_set, batch_size= batch_size, shuffle = True)

        for batch_index_test, batch_test in enumerate(data_loader_test):

            #define variables to feed the model
            samples = batch_test[0].to(device) # embedding sent to gpu
            labels = batch_test[1].to(device) # kmer class sent to gpu

            # define loss weights
            weights = []
            for item in labels :
                if int(item.data) == 0:
                    weights.append(loss_weights[0])
                if int(item.data) == 1:
                    weights.append(loss_weights[1])
            weights = torch.tensor(weights).unsqueeze(1)
            weights = weights.to(device)

            #test model
            out = model(samples)


            batch_pred = []
            #get predicted labels :
            for item in out.data:
                val = torch.round(item)
                batch_pred.append(val)
            pred.append(batch_pred.detach().cpu())

            true.append(labels.clone().detach().cpu())

            #get loss
            loss = F.binary_cross_entropy_with_logits(out,labels, weight = weights)
            testing_loss.append(loss)

            #compute precision, recall and f1
            p = precision_score(true[-1],pred[-1],labels = [0,1])
            r = recall_score(true[-1],pred[-1],labels = [0,1])
            f1 = f1_score(true[-1],pred[-1],labels = [0,1])

            # Record testing loss and metrics for each batch into the writer
            writer.add_scalar('Testing/Loss/'+event_name, loss, ts_batch_step)
            writer.add_scalars('Testing/Metrics/'+event_name,{'Precision':p,
                                                              'Recall':r,
                                                              'F1':f1},ts_batch_step)
            ts_batch_step += 1


            if batch_index_test % 2 == 0:
                print ('Epoch [{}/{}], Step [{}], Testing loss: {:.4f}'.format(epoch, num_epochs, batch_index_test, loss.item()))


        writer.flush()
    return training_loss, testing_loss, pred, true



###############################################################################
#final evaluation function (keep num_epochs = 1)
def predict_network_binary(embeddings_col, model, num_epochs, batch_size, device = torch.device('cpu'), rdseed = 13):

    torch.manual_seed(rdseed)
    np.random.seed(rdseed)
    all_pred = []
    #load dataset
    # we have to create a labels vec to init the data with KmersDataset
    #labels_col = np.zeros(len(embeddings_col))

    data =  mydatasets.predDataset(embeddings_col)# build Kmers dataset object from data

    # Define var for reporting

    pred = []
    ts_batch_step = 0
    model.eval()

    for epoch in range(1, num_epochs+1):

        data_loader_test = DataLoader(data, batch_size= batch_size, shuffle = False)

        for batch_index_test, batch_test in enumerate(data_loader_test):


            #define variables to feed the model
            samples = batch_test.to(device) # embedding sent to gpu

            #test model
            out = model(samples)

            batch_pred = []
            #get predicted labels :
            if epoch == num_epochs:
                for item in out.data:
                    val = torch.round(item)
                    batch_pred.append(val.clone().detach().cpu())
                pred.append(batch_pred)

            #pred.append(batch_pred.detach().cpu())


            ts_batch_step += 1
        all_pred.append(pred)

        # Record testing loss from each epoch into the writer

    return  all_pred
