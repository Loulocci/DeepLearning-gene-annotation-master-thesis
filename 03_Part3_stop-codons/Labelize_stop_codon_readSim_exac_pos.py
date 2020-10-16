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
parser.add_argument("-e", "--embeddings",  required=False, type=str, default = './lou/Annotation_Project/01_Part1_genic-intergenic/02_embedding/dna2vec-master/results/dna2vec-20200504-1300-k10to10-100d-10c-205Mbp-sliding-oBT-rev-WG.w2v', help = "path for dna2vec embeddings, default = 7 mers embeddings ")

# path ground truth df
parser.add_argument("-g", "--groundTruth",  required=False, type=str, default = './lou/Annotation_Project/01_Part1_genic-intergenic/Output/Ground_truth_df.csv', help = "path for ground_truth df file ")

# kmer length
parser.add_argument("-k", "--kmerLen",  required=False, type=int, default = 7, help = "Kmer length, default = 7 ")

# commit parser
args = parser.parse_args()

# on server
ground_truth_file = args.groundTruth
path_to_output = args.outputPath
reads_file = args.readsInput
embedding_file = args.embeddings
kmer_len = args.kmerLen

#load dna2vec
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

# convert insert and del lists to numeric if possible
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

# print info & example
print('There were',no_indel_count,'reads with no insertion or deletion, and',no_subs_count,'reads with no substitution, among the total of',len(reads_df),'reads.')
ground_truth.sample(1)


# ### Find all the true stop codons and potential stop codons in noisy, give index in 'Stop_codon_index'

print('Starting kmer generation \n\n')
#init columns to store start codons index
reads_df['Stop_codon_index'] = 0
reads_df['Stop_codon_index'] = reads_df['Stop_codon_index'].astype(object)

for i, read in enumerate(reads_df.Noisy_read):
    print('STEP',i,'/',len(reads_df)-1)
    # identify ground_truth genes from which start codons belong to the read
    target = ground_truth[(ground_truth.Strain_id == reads_df.Strain.loc[i]) & (ground_truth.strand == reads_df.Strand.loc[i])]

    if target.strand.mean() == 1: # forward strand
        target = target[(target.end>= reads_df.Read_start.loc[i]) & (target.end<= reads_df.Read_end.loc[i])]
        forward = True

    else: # indexing for reverse strand is inverted
        target = target[(target.end<= reads_df.Read_start.loc[i]) & (target.end>= reads_df.Read_end.loc[i])]
        forward = False

    noisy_end = []
    for end in target.end:
        if forward == True:
            end_index = end-2 - reads_df.Read_start.loc[i] #find pos in non noisy read, remove 2 because set to codon beginning
        else:
            end_index = reads_df.Read_start.loc[i] - end+2 #find pos in non noisy read, add 2 because set to codon beginning

        end_index = func.find_pos_in_noisy_read(end_index,reads_df.Indel_pos.loc[i], reads_df.Indel_size.loc[i]) #find pos in noisy read
        if end_index >= 0 : #if smaller than 0, the codon was deleted
            noisy_end.append(end_index)
        else :
            print('This stop codon was deleted in the noisy read')

    #get all possible start codons and add the actual true ones
    TAA_index = func.get_codon_index('TAA', read)
    TGA_index = func.get_codon_index('TGA', read)
    TAG_index = func.get_codon_index('TAG', read)
    reads_df['Stop_codon_index'].loc[i] =  sorted(list(set(TAA_index+TGA_index+TAG_index+noisy_end)))

# index (row index should be in order) from 0 to end
codon_kmers = func.MultiReads_to_codonKmers(reads_df.Noisy_read, model, reads_df.Stop_codon_index, kmer_size = kmer_len)

#codon_kmers_df= pd.concat([codon_kmers,genomes.head[['Strain','Strand']]], axis=1,  join='inner')
codon_kmers_df = pd.merge(codon_kmers, reads_df[['Strain','Strand','Read_start','Read_end','Indel_pos','Indel_size']], left_on='Read_id', right_index=True, how='left')

# these pos are approximative because of approximative indexing of nuc due to noise introduction, corrected in next step
codon_kmers_df['Kmer_pos_in_genome'] = -6
codon_kmers_df['Codon_pos_in_genome'] = -6
codon_kmers_df['Kmer_pos_in_genome'].loc[codon_kmers_df.Strand == 1] = codon_kmers_df['Read_start']+codon_kmers_df['Pos_kmer_in_read']
codon_kmers_df['Codon_pos_in_genome'].loc[codon_kmers_df.Strand == 1] = codon_kmers_df['Read_start']+codon_kmers_df['Pos_codon_in_read']
codon_kmers_df['Kmer_pos_in_genome'].loc[codon_kmers_df.Strand == -1] = codon_kmers_df['Read_start']-codon_kmers_df['Pos_kmer_in_read']
codon_kmers_df['Codon_pos_in_genome'].loc[codon_kmers_df.Strand == -1] = codon_kmers_df['Read_start']-codon_kmers_df['Pos_codon_in_read']

# save temporary df
now = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
codon_kmers_df.to_csv(path_to_output+'Temp_startCodons_ReadSim_'+str(kmer_len)+'mers_'+str(now))

# find the real index of the codons in the genome to avoid mislabelling
codon_kmers_df['Real_pos_codon_in_read'] = -6
codon_kmers_df['Real_pos_codon_in_genome'] = -6

for i, codon_pos in enumerate(codon_kmers_df.Pos_codon_in_read):
    print('STEP:',i,'/',len(codon_kmers_df))
    real_index = func.find_real_index(codon_pos,codon_kmers_df.Indel_pos.loc[i], codon_kmers_df.Indel_size.loc[i])
    diff = codon_pos-real_index
    if codon_kmers_df.Strand.loc[i] == 1:
        real_genome_pos = codon_kmers_df.Codon_pos_in_genome.loc[i] - diff
    else:
        real_genome_pos = codon_kmers_df.Codon_pos_in_genome.loc[i] + diff

    codon_kmers_df['Real_pos_codon_in_read'].loc[i] = real_index
    codon_kmers_df['Real_pos_codon_in_genome'].loc[i] = real_genome_pos


# ### Labelling codons as '*real start codon or not*' in the generated reads

stop_count = 0
codon_kmers_df['Is_stop_codon'] = 0
codon_kmers_df['Locus_tag'] = 0

for i in codon_kmers_df.index:
    print('STEP',i,'/',len(codon_kmers_df))
    kmer = codon_kmers_df.loc[i]
    # check if same strain
    target = ground_truth[ground_truth['Strain_id'] == kmer.Strain]
    # check if same strand
    target = target[target['strand'] == kmer.Strand]
    #check if the kmer has an actual start in the target
    if kmer.Strand == 1: # add or remove 2 to set pos to beginning of stop codon
        target = target[(target['end']-2<= kmer.Real_pos_codon_in_genome+1) & (target['end']-2>= kmer.Real_pos_codon_in_genome-1)]
    else:
        # the codon start is located at the biggest numerical extremity of the gene in reverse strand
        target = target[(target['start']+2<= kmer.Real_pos_codon_in_genome+1) & (target['start']+2>= kmer.Real_pos_codon_in_genome-1)]
    #check if something found
    if (target.empty == False):
        stop_count += 1
        # if there is more than one match
        if len(target) > 1:
            print('More than one match!')
        else :
            # if something found codon labelled as codon start
            print('A stop codon found!')
            codon_kmers_df.Is_stop_codon.loc[i] = 1
            try :
                codon_kmers_df.Locus_tag.loc[i] = target.locus_tag.values[0]
            except :
                codon_kmers_df.Locus_tag.loc[i] = 0


# save output df
now = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
codon_kmers_df.to_csv(path_to_output+'startCodons_ReadSim_'+str(kmer_len)+'mers_'+str(now))

# print info
print(len(codon_kmers_df[codon_kmers_df['Is_stop_codon']==1]), 'stop codons identified!')
print('NC_000913.3 forward :',len(codon_kmers_df.Codon_pos_in_genome[(codon_kmers_df['Is_stop_codon'] == 1) & (codon_kmers_df['Strand']== 1) & (codon_kmers_df['Strain']== 'NC_000913.3')].unique()))
print('NC_000913.3 reverse :',len(codon_kmers_df.Codon_pos_in_genome[(codon_kmers_df['Is_stop_codon'] == 1) & (codon_kmers_df['Strand']== -1) & (codon_kmers_df['Strain']== 'NC_000913.3')].unique()))

print('NC_011751.1 forward :',len(codon_kmers_df.Codon_pos_in_genome[(codon_kmers_df['Is_stop_codon'] == 1) & (codon_kmers_df['Strand']== 1) & (codon_kmers_df['Strain']== 'NC_011751.1')].unique()))
print('NC_011751.1 reverse :',len(codon_kmers_df.Codon_pos_in_genome[(codon_kmers_df['Is_stop_codon'] == 1) & (codon_kmers_df['Strand']== -1) & (codon_kmers_df['Strain']== 'NC_011751.1')].unique()))

print('NC_018658.1 forward :',len(codon_kmers_df.Codon_pos_in_genome[(codon_kmers_df['Is_stop_codon'] == 1) & (codon_kmers_df['Strand']== 1) & (codon_kmers_df['Strain']== 'NC_018658.1')].unique()))
print('NC_018658.1 reverse :',len(codon_kmers_df.Codon_pos_in_genome[(codon_kmers_df['Is_stop_codon'] == 1) & (codon_kmers_df['Strand']== -1) & (codon_kmers_df['Strain']== 'NC_018658.1')].unique()))

print('NC_002695.2 forward :',len(codon_kmers_df.Codon_pos_in_genome[(codon_kmers_df['Is_stop_codon'] == 1) & (codon_kmers_df['Strand']== 1) & (codon_kmers_df['Strain']== 'NC_002695.2')].unique()))
print('NC_002695.2 reverse :',len(codon_kmers_df.Codon_pos_in_genome[(codon_kmers_df['Is_stop_codon'] == 1) & (codon_kmers_df['Strand']== -1) & (codon_kmers_df['Strain']== 'NC_002695.2')].unique()))
