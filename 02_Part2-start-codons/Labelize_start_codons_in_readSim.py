#!/usr/bin/env python
# coding: utf-8

# This script allows to generate and give labels to start codon k-mers. Here the approach is zone targeted,
# it means that we generate k-mers only from DNA regions containing a potential start codon (ATG, GTG, TTG codons)

#IMPORTS

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
parser = argparse.ArgumentParser(description="NN run config")

# path to reads df input
parser.add_argument("-i", "--readsInput",  required=True, type=str, help = "file for reads dataframe")

# path to output
parser.add_argument("-o", "--outputPath",  required=False, type=str, default = './', help = "path for output, default = './' ")

# path to dna2vec embeddings
parser.add_argument("-e", "--embeddings",  required=False, type=str, default = '/tudelft.net/staff-umbrella/abeellab/lou/Annotation_Project/01_Part1_genic-intergenic/02_embedding/dna2vec-master/results/dna2vec-20200504-1300-k10to10-100d-10c-205Mbp-sliding-oBT-rev-WG.w2v', help = "path for dna2vec embeddings, default = 10 mers embeddings ")

# path ground truth df
parser.add_argument("-g", "--groundTruth",  required=False, type=str, default = '/tudelft.net/staff-umbrella/abeellab/lou/Annotation_Project/01_Part1_genic-intergenic/03_Generate_kmer/Output/Ground_truth_df.csv', help = "path for dna2vec embeddings, default = 4 Strains Ecoli ")

# kmer length
parser.add_argument("-k", "--kmerLen",  required=False, type=int, default = 10, help = "Kmer length, default = 10 ")

# window around codon position
parser.add_argument("-w", "--window",  required=False, type=int, default = 10, help = "Window of position around the codon position, accepted as start position, default = 10 ")

# commit parser
args = parser.parse_args()


# on server
ground_truth_file = args.groundTruth
path_to_output = args.outputPath
reads_file = args.readsInput
embedding_file = args.embeddings
kmer_len = args.kmerLen
window = args.window

# load ground truth
ground_truth = pd.read_csv(ground_truth_file)

#load dna2vec
model = wk.load_word2vec_format(embedding_file, binary = False)

reads_df = pd.read_csv(reads_file).drop('Unnamed: 0',axis = 1)
reads_df.Strain.loc[reads_df.Strain == 'CP003289.1'] = 'NC_018658.1'


# ### Generate Kmers
print('Starting kmer generation \n\n')
#init columns to store start codons index
reads_df['Start_codon_index'] = 0
reads_df['Start_codon_index'] = reads_df['Start_codon_index'].astype(object)


for i, read in enumerate(reads_df.Noisy_read):
    print('STEP',i,'/',len(reads_df)-1)

    ATG_index = func.get_codon_index('ATG', read)
    GTG_index = func.get_codon_index('GTG', read)
    TTG_index = func.get_codon_index('TTG', read)
    reads_df['Start_codon_index'].loc[i] = ATG_index+GTG_index+TTG_index


# index (row index should be in order) from 0 to end sinon pb
codon_kmers = func.MultiReads_to_codonKmers(reads_df.Noisy_read, model, reads_df.Start_codon_index, kmer_size = kmer_len)


codon_kmers_df= pd.concat([codon_kmers,genomes.head[['Strain','Strand']]], axis=1,  join='inner')
codon_kmers_df = pd.merge(codon_kmers, reads_df[['Strain','Strand','Read_start','Read_end']], left_on='Read_id', right_index=True, how='left')

# these pos are approximative because of approximative indexing of nuc due to noise introduction
codon_kmers_df['Kmer_pos_in_genome'] = np.nan
codon_kmers_df['Codon_pos_in_genome'] = np.nan
codon_kmers_df['Kmer_pos_in_genome'].loc[codon_kmers_df.Strand == 1] = codon_kmers_df['Read_start']+codon_kmers_df['Pos_kmer_in_read']
codon_kmers_df['Codon_pos_in_genome'].loc[codon_kmers_df.Strand == 1] = codon_kmers_df['Read_start']+codon_kmers_df['Pos_codon_in_read']
codon_kmers_df['Kmer_pos_in_genome'].loc[codon_kmers_df.Strand == -1] = codon_kmers_df['Read_start']-codon_kmers_df['Pos_kmer_in_read']
codon_kmers_df['Codon_pos_in_genome'].loc[codon_kmers_df.Strand == -1] = codon_kmers_df['Read_start']-codon_kmers_df['Pos_codon_in_read']


# ### Labelling codons as 'real start codon or not' in the generated reads
start_count = 0
codon_kmers_df['Is_start_codon'] = 0
codon_kmers_df['Locus_tag'] = 0

for i in codon_kmers_df.index:
    print('STEP',i,'/',len(codon_kmers_df))
    kmer = codon_kmers_df.loc[i]
    # check if same strain
    target = ground_truth[ground_truth['Strain_id'] == kmer.Strain]
    # check if same strand
    target = target[target['strand'] == kmer.Strand]
    #check if the kmer has an actual start in the target
    if kmer.Strand == 1:
        target = target[(target['start']<= kmer.Codon_pos_in_genome+window) & (target['start']>= kmer.Codon_pos_in_genome-window)]
    else:
        # the codon start is located at the biggest numerical extremity of the gene in reverse strand
        target = target[(target['end']<= kmer.Codon_pos_in_genome+window) & (target['end']>= kmer.Codon_pos_in_genome-window)]
    #check if something found
    if (target.empty == False):
        start_count += 1
        # if there is more than one match
        if len(target) > 1:
            print('More than one match!')
        else :
            # if something found codon labelled as codon start
            print('A start codon found!')
            codon_kmers_df.Is_start_codon.loc[i] = 1
            try :
                codon_kmers_df.Locus_tag.loc[i] = target.locus_tag.values[0]
            except :
                codon_kmers_df.Locus_tag.loc[i] = 0


#Save output
now = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
codon_kmers_df.to_csv(path_to_output+'StartCodons_ReadSim_'+str(kmer_len)+'mers_+-'+str(window)+'_'+str(now))

# print stats and info
print(len(codon_kmers_df[codon_kmers_df['Is_start_codon']==1]), 'start codons identified!')
print('NC_000913.3 forward :',len(codon_kmers_df.Codon_pos_in_genome[(codon_kmers_df['Is_start_codon'] == 1) & (codon_kmers_df['Strand']== 1) & (codon_kmers_df['Strain']== 'NC_000913.3')].unique()))
print('NC_000913.3 reverse :',len(codon_kmers_df.Codon_pos_in_genome[(codon_kmers_df['Is_start_codon'] == 1) & (codon_kmers_df['Strand']== -1) & (codon_kmers_df['Strain']== 'NC_000913.3')].unique()))

print('NC_011751.1 forward :',len(codon_kmers_df.Codon_pos_in_genome[(codon_kmers_df['Is_start_codon'] == 1) & (codon_kmers_df['Strand']== 1) & (codon_kmers_df['Strain']== 'NC_011751.1')].unique()))
print('NC_011751.1 reverse :',len(codon_kmers_df.Codon_pos_in_genome[(codon_kmers_df['Is_start_codon'] == 1) & (codon_kmers_df['Strand']== -1) & (codon_kmers_df['Strain']== 'NC_011751.1')].unique()))

print('NC_018658.1 forward :',len(codon_kmers_df.Codon_pos_in_genome[(codon_kmers_df['Is_start_codon'] == 1) & (codon_kmers_df['Strand']== 1) & (codon_kmers_df['Strain']== 'CP003289.1')].unique()))
print('NC_018658.1 reverse :',len(codon_kmers_df.Codon_pos_in_genome[(codon_kmers_df['Is_start_codon'] == 1) & (codon_kmers_df['Strand']== -1) & (codon_kmers_df['Strain']== 'NC_018658.1')].unique()))

print('NC_002695.2 forward :',len(codon_kmers_df.Codon_pos_in_genome[(codon_kmers_df['Is_start_codon'] == 1) & (codon_kmers_df['Strand']== 1) & (codon_kmers_df['Strain']== 'NC_002695.2')].unique()))
print('NC_002695.2 reverse :',len(codon_kmers_df.Codon_pos_in_genome[(codon_kmers_df['Is_start_codon'] == 1) & (codon_kmers_df['Strand']== -1) & (codon_kmers_df['Strain']== 'NC_002695.2')].unique()))
