#!/usr/bin/env python
# coding: utf-8

# This script fixes the mislabelling problem and improves the some previous kmers dataframe (prot tag etc.)

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
import statistics
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

# argument parser
parser = argparse.ArgumentParser(description="Labelize readSim corrected for pos")

# path to reads df input
parser.add_argument("-i", "--kmersInput",  required=True, type=str, help = "file for reads dataframe")

# path to output
parser.add_argument("-o", "--outputPath",  required=False, type=str, default = './', help = "path for output, default = './' ")

# path ground truth df
parser.add_argument("-g", "--groundTruth",  required=False, type=str, default = './lou/Annotation_Project/Output/Ground_truth_df.csv', help = "path for ground_truth df file ")


# commit parser
args = parser.parse_args()


# ### PREPARE DATA

# arguments
ground_truth_file = args.groundTruth
path_to_output = args.outputPath
kmers_file = args.kmersInput

#load files
ground_truth = pd.read_csv(ground_truth_file)
kmers_df = pd.read_csv(kmers_file)
kmers_df.drop(['Unnamed: 0'],axis =1, inplace = True)

# convert string array to list
kmers_df.Indel_pos = kmers_df.Indel_pos.str.strip('[ ]')
kmers_df.Indel_pos = kmers_df.Indel_pos.str.split(',')
kmers_df.Indel_size = kmers_df.Indel_size.str.strip('[]')
kmers_df.Indel_size = kmers_df.Indel_size.str.split(',')

kmers_df.Subs_pos = kmers_df.Subs_pos.str.strip('[ ]')
kmers_df.Subs_pos = kmers_df.Subs_pos.str.split(',')
kmers_df.Subs_size = kmers_df.Subs_size.str.strip('[]')
kmers_df.Subs_size = kmers_df.Subs_size.str.split(',')


# find the real index of the codons in the genome to avoid mislabelling
kmers_df['Real_start_in_read'] = np.nan
max_diff = 0
for idx, kmer_pos in enumerate(kmers_df.Start_in_read):
    print('STEP:',idx,'/',len(kmers_df))
    print('Kmer pos in read:',kmer_pos)
    pos_toint= [int(i) for i in kmers_df.Indel_pos[kmers_df.Read_id == kmers_df.Read_id.loc[idx]].iloc[0]]
    size_toint= [int(i) for i in kmers_df.Indel_size[kmers_df.Read_id == kmers_df.Read_id.loc[idx]].iloc[0]]
    real_index = func.find_real_index(kmer_pos, pos_toint, size_toint)
    print('Real index:',real_index)
    if (real_index >= 0) and (real_index < kmers_df.Noisy_read_len.loc[idx]):
        kmers_df['Real_start_in_read'].loc[idx] = real_index

    else:
        #real_genome_pos = np.nan
        print('NaN kmer because real index :',real_index,'and read len:',kmers_df.Noisy_read_len.loc[idx] )
        kmers_df['Real_start_in_read'].loc[idx] = np.nan


kmers_df['Kmer_pos_in_genome'] = np.nan
kmers_df['Kmer_pos_in_genome'].loc[kmers_df.Strand == 1] = kmers_df.Noisy_read_start+kmers_df.Start_in_read
kmers_df['Kmer_pos_in_genome'].loc[kmers_df.Strand == -1] = kmers_df.Noisy_read_start-kmers_df.Start_in_read

#show a sample
kmers_df.sample(2)


# ### START LABELIZE

kmers_df['Genic'] = np.nan
kmers_df['Locus_tag'] = np.nan

for i in kmers_df.index:
    print('Step:',i,'/',len(kmers_df.index))
    my_kmer = kmers_df.loc[i]
    # check if same strain
    target = ground_truth[ground_truth['Strain_id'] == my_kmer.Strain]
    # check if same strand
    target = target[target['strand'] == my_kmer.Strand]
    #check if kmer_start in between CDS start and end in target
    if my_kmer.Strand == 1:
        target = target[(target['start']<= my_kmer.Kmer_pos_in_genome)&(target['end']>= my_kmer.Kmer_pos_in_genome)]
    else:
        target = target[(target['start']>= my_kmer.Kmer_pos_in_genome)&(target['end']<= my_kmer.Kmer_pos_in_genome)]
    #check if something is found
    if (target.empty == False):
        if len(target) ==1 :
            kmers_df['Genic'].loc[i] = True
            kmers_df['Locus_tag'].loc[i] = ground_truth.locus_tag.loc[target.index.item()]
        else:
            locus =str(ground_truth.locus_tag.loc[target.index[0]])
            for row in target.index[1:]:
                locus+=', '+ground_truth.locus_tag.loc[row]


    else:
        kmers_df['Genic'].loc[i] = False

# print final info
print('Percentage of coding regions in the',len(kmers_df.Read_id.unique()),'reads dataset:',(len(kmers_df[kmers_df.Genic == True])/len(kmers_df))*100)
random_read = random.randrange(0,max(kmers_df.Read_id.unique()),1)
print('Here is a first example: read composition for the random read',random_read,':')
print('Strain:',kmers_df.Strain[(kmers_df.Read_id == random_read)].unique().item(),', Strand:',kmers_df.Strand[(kmers_df.Read_id == random_read)].unique().item(),', Locus found on that read:')
print(kmers_df.Locus_tag[(kmers_df.Read_id == random_read) & (kmers_df.Genic == True)].unique())

random_read = random.randrange(0,max(kmers_df.Read_id.unique()),1)
print('\nHere is a second example: read composition for the random read',random_read,':')
print('Strain:',kmers_df.Strain[(kmers_df.Read_id == random_read)].unique().item(),', Strand:',kmers_df.Strand[(kmers_df.Read_id == random_read)].unique().item(),', Locus found on that read:')
print(kmers_df.Locus_tag[(kmers_df.Read_id == random_read) & (kmers_df.Genic == True)].unique())

#save output
now = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
kmers_df.to_csv(path_to_output+'WholeReadKmers_ReadSim_exactPos_'+str(kmer_len)+'mers_'+str(now)+'_labelled')
