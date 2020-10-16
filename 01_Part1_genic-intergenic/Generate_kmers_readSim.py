#!/usr/bin/env python
# coding: utf-8

#IMPORTS

#Basics#Basics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy import stats
import locale
locale.setlocale(locale.LC_ALL, '')
import datetime
import random
import argparse

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
#word2vec
import gensim.models.keyedvectors as word2vec
from gensim.models.keyedvectors import KeyedVectors as wk

# argument parser
parser = argparse.ArgumentParser(description="Labelize readSim corrected for pos")

# path to reads df input
parser.add_argument("-i", "--readsInput",  required=True, type=str, help = "file for reads dataframe")

# path to output
parser.add_argument("-o", "--outputPath",  required=False, type=str, default = './', help = "path for output, default = './' ")

# path to dna2vec embeddings
parser.add_argument("-e", "--embeddings",  required=False, type=str, default = './lou/Annotation_Project/01_Part1_genic-intergenic/02_embedding/dna2vec-master/results/dna2vec-20200504-1300-k10to10-100d-10c-205Mbp-sliding-oBT-rev-WG.w2v', help = "path for dna2vec embeddings, default = 10 mers embeddings ")

# path ground truth df
parser.add_argument("-g", "--groundTruth",  required=False, type=str, default = './lou/Annotation_Project/01_Part1_genic-intergenic/Output/Ground_truth_df.csv', help = "path for ground_truth df file ")

# kmer length
parser.add_argument("-k", "--kmerLen",  required=False, type=int, default = 7, help = "Kmer length, default = 7 ")

# commit parser
args = parser.parse_args()

# setting arguments
ground_truth_file = args.groundTruth
path_to_output = args.outputPath
reads_file = args.readsInput
embedding_file = args.embeddings
kmer_len = args.kmerLen

#load dna2vec (embedding model)
model = wk.load_word2vec_format(embedding_file, binary = False)

# load data
ground_truth = pd.read_csv(ground_truth_file)
reads_df = pd.read_csv(reads_file).drop('Unnamed: 0',axis = 1)
reads_df.Strain.loc[reads_df.Strain == 'CP003289.1'] = 'NC_018658.1'

# convert string array to list
reads_df.Indel_pos = reads_df.Indel_pos.str.strip('[ ]')
reads_df.Indel_pos = reads_df.Indel_pos.str.split(',')
reads_df.Indel_size = reads_df.Indel_size.str.strip('[]')
reads_df.Indel_size = reads_df.Indel_size.str.split(',')

reads_df.Subs_pos = reads_df.Subs_pos.str.strip('[ ]')
reads_df.Subs_pos = reads_df.Subs_pos.str.split(',')
reads_df.Subs_size = reads_df.Subs_size.str.strip('[]')
reads_df.Subs_size = reads_df.Subs_size.str.split(',')


# convert insertion and deletion index lists to numeric if possible
no_indel_count = 0
no_subs_count = 0
for idx in reads_df.index:
    print('Step:',idx,'/',reads_df.index[-1])
    try:
        reads_df.Indel_pos.loc[idx] = [int(i) for i in reads_df.Indel_pos.loc[idx]]
    except ValueError:
        print('No insertion or deletion for this read:', idx)
        no_indel_count += 1
        reads_df.Indel_pos.loc[idx] ='nan'
    try:
        reads_df.Indel_size.loc[idx] = [int(i) for i in reads_df.Indel_size.loc[idx]]
    except ValueError:
        reads_df.Indel_size.loc[idx] ='nan'

    try:
        reads_df.Subs_pos.loc[idx] = [int(i) for i in reads_df.Subs_pos.loc[idx]]
    except ValueError:
        print('No substitution for this read:', idx)
        no_subs_count += 1
        reads_df.Subs_pos.loc[idx] ='nan'
    try:
        reads_df.Subs_size.loc[idx] = [int(i) for i in reads_df.Subs_size.loc[idx]]
    except ValueError:
        reads_df.Subs_size.loc[idx] ='nan'

#print info
print('There were',no_indel_count,'reads with no insertion or deletion, and',no_subs_count,'reads with no substitution, among the total of',len(reads_df),'reads.')

#Define kmers df
kmers = func.Multiseq_to_vector(reads_df.Noisy_read, model, kmer_size = args.kmerLen)
kmers_df = pd.merge(kmers, reads_df[['Strain','Strand','Read_start','Read_end','Indel_pos','Indel_size','Subs_pos','Subs_size']], left_on='Read_id', right_index=True, how='left')

#save output
now = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
kmers_df.to_csv(path_to_output+'WholeReadKmers_ReadSim_exactPos_'+str(kmer_len)+'mers_'+str(now))
