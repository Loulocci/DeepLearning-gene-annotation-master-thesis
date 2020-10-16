#!/usr/bin/env python
# coding: utf-8

# In[1]:
#######IMPORTS#####################
#basics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import seaborn as sns
from scipy import stats

#Biopython
from Bio.Align import MultipleSeqAlignment
from Bio import SeqIO
from Bio.pairwise2 import format_alignment
from Bio import pairwise2
from Bio import Entrez
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, FeatureLocation
import random

#Deep learning
from enum import IntEnum
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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


####################################################
#### PREPROCESSING FUNCTIONS
####################################################

#### GROUND TRUTH FUNCTIONS

#Function found following :  https://dmnfarrell.github.io/bioinformatics/genbank-python, and customized
#This function converts genbank files into pandas dataframe
#pandas and Biopython are required

#INPUT: It takes a genbank filename as input (make sure to indicate the good path)
#OUTPUT: the output is a dataframe containing all the features

def features_to_dataframe(gb_file):

    genome = SeqIO.read(open(gb_file,"r"), "genbank")
    #preprocess features
    allfeat = []
    # item = index of the feature, f = the feature
    for (item, f) in enumerate(genome.features):

        #x contains : location, type, qualifiers
        x = f.__dict__
        #q = qualifiers, contained in x, and contains :
        q = f.qualifiers
        #updates adds an element to a dictionnary if not already in the dict
        x.update(q)
        featurekeys = q.keys()

        # d is the final dictionnary containing all the features
        d = {}
        d['start'] = f.location.start
        d['end'] = f.location.end
        d['strand'] = f.location.strand
        d['type'] = f.type

        for i in featurekeys:
            if i in x:
                if type(x[i]) is list:
                    d[i] = x[i][0]
                else:
                    d[i] = x[i]
        allfeat.append(d)

        #update featurekeys to the final dict d.
        featurekeys= d.keys()

    df = pd.DataFrame(allfeat,columns=featurekeys)

    if len(df) == 0:
        print ('ERROR: genbank file return empty data, check that the file contains protein sequences '               'in the translation qualifier of each protein feature.')
    return df

####
#converts .paf file into a pandas dataframe. 12 columns described at : https://github.com/lh3/miniasm/blob/master/PAF.md

#INPUT: .paf file (TAB delimiter)
#OUTPUT: pandas Dataframe of the data contained in the .paf file. Be carefull, the optional
# SAM-like typed key-value pairs is not included in the datframe.

def paf_to_dataframe(paf_file):
    col_names = ['Seq_name','Seq_len','Start','End','Strand','Target_seq_name','Target_seq_len','Target_start',
                 'Target_end','Nb_residue_match','Align_block_len','Mapping_quality']

    df = pd.read_csv(paf_file, header=None, sep = '\t')
    mapping_df = df.loc[:,0:11]
    mapping_df.columns = col_names
    return mapping_df


#### EMBEDDING FUNCTIONS

# This function convert an input dna seq into its corresponding dna2vec embedding
#
# INPUT : seq = string dna sequence, model = .w2v model loaded trough word2vec (model= wk.load_word2vec_format(path2file, binary = False))
#kmer_size = size of the kmers for this seq, rand_kmer_size = True if we use random kmer size,
#kmer_size_upper and lower = define boudaries of kmer size, shift = window shift to get kmers
#
# OUTPUT : embed_vec = vector of vector, containing the embedding vector of each kmer in the seq.
# kmer_vec = vector of kmers generated

def Seq_to_vector(seq, model, kmer_size = 7, rand_kmer_size = False, kmer_size_lower = 0, kmer_size_upper = 0, shift = 1) :

    if (shift > kmer_size and shift > kmer_size_upper or shift <1 or not(isinstance(shift, int))):
        raise Exception('Error: wrong value for shift')

    kmer_vec = []
    embed_vec = []
    for nuc in range(0,len(seq), shift) :
        if (rand_kmer_size == True):
            if (kmer_size_upper<1 or kmer_size_lower <0 or not(isinstance(kmer_size_lower, int)) or not(isinstance(kmer_size_lower, int))):
                raise Exception('Error: wrong value for kmer size bounds')

            kmer_size = -1
            kmer_len= np.random.randint(kmer_size_lower, high=kmer_size_upper)
        else:
            kmer_len = kmer_size

        kmer = seq[nuc:nuc+kmer_len]


        #ignore kmer with a too small size
        if (len(kmer) <= kmer_size_upper and len(kmer) >= kmer_size_lower or len(kmer) == kmer_size):
            #append embedding vector corresponding to the kmer
            embed_vec.append(model.get_vector(kmer))
            #save kmers in a separated vector
            kmer_vec.append(kmer)

    #returns a vector containing embedding and a vector containing kmers
    return embed_vec, kmer_vec


####
# This function convert an input set of dna sequences into its corresponding dna2vec embedding

# INPUT : seq = string dna sequence, model = .w2v model loaded trough word2vec (model= wk.load_word2vec_format(path2file, binary = False))
# kmer_size = size of the kmers for this seq, rand_kmer_size = True if we use random kmer size,
# kmer_size_upper and lower = define boudaries of kmer size, shift = window shift to get kmers
#
# OUTPUT : returns a df containing
#embed_vec = vector of vector, containing the embedding vector of each kmer for each seq.
# kmer_vec = vector of kmers generated
# pos_in_read = position of the kmer in read
#read_id = gives the list of generated read_ids
#start_in_read = kmer start pos in read
def Multiseq_to_vector(sequences, model, kmer_size = 7, rand_kmer_size = False, kmer_size_lower = 0, kmer_size_upper = 0, shift = 1):

    vector = []
    kmers = []
    pos_in_read = []
    read_id = []
    start = []
    counter = 0
    for seq in sequences :
        print('Progress : ',counter,'/',len(sequences))
        embed_vec, kmer_vec = Seq_to_vector(seq,model,kmer_size, rand_kmer_size,kmer_size_lower,kmer_size_upper, shift)
        vector = vector+embed_vec
        kmers = kmers+kmer_vec
        pos_in_read = pos_in_read+list(range(len(kmer_vec)))
        read_id = read_id+list(np.full((len(kmer_vec)),counter, dtype=int)) #add a list of counter, of size len(kmer_vec)
        counter += 1

        # find kmer start in read
        i = 1
        kmer_start = [0]
        for kmer in kmer_vec:
            kmer_start.append(kmer_start[i-1]+shift)
            i += 1
        kmer_start.pop(); #remove last element, since there is no next kmer
        start = start+kmer_start


    #returns a vector containing embedding and a vector containing kmers
    output_df = pd.DataFrame({'Read_id': read_id,'Embedding': vector, 'Kmers': kmers,'Pos_in_read': pos_in_read,'Start_in_read': start}, columns=['Read_id', 'Embedding','Kmers','Pos_in_read','Start_in_read'])
    return output_df

####
# This function extract kmers containing a certain codon from an input dna read

# INPUT : read = string dna sequence, model = .w2v model loaded trough word2vec (model= wk.load_word2vec_format(path2file, binary = False))
# kmer_size = size of the kmers for this seq, codon_index = array containing indexes for codons of interest
#
#
# OUTPUT : returns several vectors:
#embed_vec = vector of vector, containing the embedding vector of each kmer for each seq.
# kmer_vec = vector of kmers generated
# kmers_pos = start position of the kmer in read

def Read_to_codonKmers(read, model, codon_index, kmer_size = 7) :


    kmer_vec = []
    embed_vec = []
    kmer_pos = []
    for index in codon_index :

        # generate a random number for the window of each kmer (where the start codon will be placed in the kmer)
        pos = random.randint(0,kmer_size -3)
        # check if window out of bound
        if index + kmer_size - pos < len(read) and index-pos >= 0 :
            kmer = read[index-pos:index + kmer_size - pos]
        # adapt when codons are located at extremities of the reads
        #if codon close to end of read
        elif index + kmer_size - pos > len(read):
            pos = kmer_size -3
            kmer  = read[index-pos:index + kmer_size - pos]
        #if codon close top beginning of read
        else :
            pos = index
            kmer  = read[0:kmer_size]

        try:
            embed_vec.append(model.get_vector(kmer))
            kmer_vec.append(kmer)
            # keep index of kmer's start in the read
            kmer_pos.append(index-pos)
        except:
            print('kmer',kmer,'not in vocabulary')
            embed_vec.append('None')
            kmer_vec.append(kmer)
            kmer_pos.append(index-pos)



    #returns a vector containing embedding and a vector containing kmers
    return kmer_vec, embed_vec, kmer_pos

####
# This function calls the previous function in loop and produces a final dataframe containing codon_kmers and their position in reads

# INPUT : read = string dna sequence, model = .w2v model loaded trough word2vec (model= wk.load_word2vec_format(path2file, binary = False))
# kmer_size = size of the kmers for this seq, codon_index = array containing indexes for codons of interest
#
#
# OUTPUT : returns a df containing
# Read_id = arbitrary read ID (same ID for kmers in the same read),
# Embedding = codon_kmer corresponding embedding vector
# Kmers = the codon kmer seq
#Pos_kmer_in_read = start of the kmer in read
# Pos_codon_in_read = start of the codon in read

def MultiReads_to_codonKmers(reads, model, codon_index, kmer_size = 7):

    embeddings = []
    kmers = []
    kmers_start = []
    pos_codon = []
    read_id = []

    for i, read in enumerate(reads):
        print('Progress : ',i,'/',len(reads))

        kmer_vec, embed_vec , kmer_pos = Read_to_codonKmers(read, model, codon_index[i], kmer_size = kmer_size)
        embeddings = embeddings+embed_vec
        kmers = kmers+kmer_vec
        kmers_start = kmers_start+kmer_pos
        read_id = read_id+list(np.full((len(kmer_vec)),i, dtype=int)) #add a list of counter, of size len(kmer_vec)
        pos_codon = pos_codon+codon_index[i]


    #returns a vector containing embedding and a vector containing kmers
    output_df = pd.DataFrame({'Read_id': read_id,'Embedding': embeddings, 'Kmers': kmers,'Pos_kmer_in_read': kmers_start, 'Pos_codon_in_read':pos_codon}, columns=['Read_id', 'Embedding','Kmers','Pos_kmer_in_read','Pos_codon_in_read'])
    return output_df


####
# This function get the indexes of all occurence of a specified codon in a read
# INPUT : codon = string (ex: 'ATG'), seq = string read (ex : 'CCTTTTGGGTTGGTAAAGAGAGTCGACGTAAAG..')
#
#
# OUTPUT : returns an array containing the indexes

def get_codon_index(codon, seq):
    i = 0
    codon_index = []
    while codon in seq[i:len(seq)]:
        codon_index.append(seq.index(codon,i))
        i = seq.index(codon, i)+1

    return codon_index


###
# this function gives the real index of a nuc in genome depending on the noise introduced.
#INPUT: index of the nucleotide in the noisy read, list of position modified, list of size (and type depending on the signe) of modification
# OUTPUT: index in genome. index -2 means that this nuc was inserted, and actually does not exist in genome
def find_real_index(index,indel_pos, indel_size):

    for i,pos in reversed(list(enumerate(indel_pos))) :

        if index > -2 :
            if indel_size != 'nan':
                if indel_size[i] > 0: # check if previous modif was insertion
                    if pos < index : # look at position of the insertion
                        if pos+indel_size[i] < index : # look if index part of inserted nuc
                            index = index - indel_size[i] # index is not part of inserted nuc
                        else:
                            index = -2 # index is part of inserted nuc
                if indel_size[i] < 0:# check if previous modif was deletion
                    if pos <= index:# check if index after del
                        index = index - indel_size[i] # if it is then change index


    return index


###
# this function gives the final index of a nuc in the read after introduction of noise.
#INPUT: index of the nucleotide in the read before noise introduction, list of position modified, list of size (and type depending on the signe) of modification
# OUTPUT: index in read. index -2 means that this nuc was inserted, and actually does not exist in genome
def find_pos_in_noisy_read(gt_index,indel_pos, indel_size):

    for i, pos in enumerate(indel_pos) :
        if gt_index > -1 :
            if indel_size != 'nan':
                if indel_size[i] > 0: # if modif was insertion
                    if pos < gt_index : # look at position of the insertion
                        gt_index = gt_index + indel_size[i] # it modifies the index
                if indel_size[i] < 0:# if modif was deletion
                    if pos <= gt_index:# if deletion occurs before gt_index
                        if pos+abs(indel_size[i]) > gt_index :
                            gt_index = -2 # this position was deleted
                        else:
                            gt_index = gt_index + indel_size[i] #it modifies the index


    return gt_index


####################################################
#### METRICS FUNCTIONS
####################################################

####
# INPUT : takes model output array for froward strand and for reverse Strand
# OUTPUT : output corresponding classe (0 = non genic, 1 genic forward, -1 genic reverse, 2 = genic both)
## get class of the output or label
def get_class(forward,reverse):
    out = 99
    f = int(round(forward))
    r = int(round(reverse))
    if ((f == 1) and (r == 0)):
        out = 1
    elif ((f == 0) and (r == 1)):
        out = -1
    elif ((f == 1) and (r == 1)):
        out = 2
    elif ((f == 0) and (r == 0)):
        out = 0

    return out

####
## get confusion matrix for 2dimension output
def confusion_matrix_multiclass(out,labels, class_order):
    size = len(out)
    pred = []
    for i in range(size):
        val,index = torch.max(out[i],0)
        pred.append(index.data.item())

    return multilabel_confusion_matrix(labels,pred,labels = class_order)



####################################################
#### SCORING PART FUNCTIONS
####################################################

####
#This functions finds the scores for start and stops. It looks at the 12 kmers around
# the start or stop codon and sums the predictions
def get_scores(prediction, idx):
    allscores = []

    for i in idx:
        print('Step:',i)
        if i >=6 and i<=len(prediction)-6:#len prediction is equal to read len -6
            kmer= prediction[i-6:i+6]
            score = sum(kmer)/len(kmer)
            allscores.append(score)
        elif i <6: #kmer close to beginning of read
            kmer= prediction[0:i+6]
            score = sum(kmer)/len(kmer)
            allscores.append(score)
        elif i > len(prediction)-6: #kmer close to end of read
            kmer= prediction[i-6:len(prediction)]
            score = sum(kmer)/len(kmer)
            allscores.append(score)

    return allscores

####
#This function finds all the codons in frame in the same read, it means distant from multiple of 3
# at least distant of 150 bp, max of 5000bp and in the right order depending of the strand direction
def find_in_frame_codons(start_vec, stop_vec, strand, min_dist = 0, max_dist = 5000):
    inframe = []
    for start in start_vec:
        for stop in stop_vec:
            # check if stop after start depending on the strand (1 for forward, -1 for reverse)
            if strand*(stop-start) > 0:
                #check if distant from a multiple of 3
                if abs(stop-start)%3 == 0:
                    couple = (start,stop)
                    #check if distance in between them is enough
                    if (abs(stop-start)>= min_dist) and (abs(stop-start)<= max_dist):
                        inframe.append(couple)

    return inframe

###
#this function gives the genic score with adding all the prediction in between the 2 codons given as couple
def compute_genic_score(couple, prediction, strand):
    score = 0
    len_candidate = abs(couple[0]-couple[1])

    for i in range(couple[0],couple[1]+1):
        score+= prediction[i]
    score = score/len_candidate


    return score


###
# say if 2 candidate genes are overlapping or not
def are_overlapping(c1,c2):
    c1_start = c1[0]
    c1_stop = c1[1]
    c2_start = c2[0]
    c2_stop = c2[1]
    overlap = False
    if ((c1_start>= c2_start) and (c1_start<=c2_stop)) or ((c2_start>= c1_start) and (c2_start<= c1_stop)) or ((c1_stop>= c2_start) and (c1_stop<= c2_stop)) or ((c2_stop>= c1_start) and (c2_stop<= c1_stop)):
        overlap = True

    return overlap


###
#give all the candidate gene overlapping in a list of candidates defined by their start and stop couple position
def find_overlapping_candidates(target,couples,scores):
    overlap = []
    overlap_scores = []
    for i,candidate in enumerate(couples):
        if target != candidate:
            if are_overlapping(target,candidate):
                overlap.append(candidate)
                overlap_scores.append(scores[i])


    return overlap, overlap_scores
