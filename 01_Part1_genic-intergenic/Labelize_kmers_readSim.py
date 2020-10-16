#!/usr/bin/env python
# coding: utf-8

# ## Find Labels for kmers
#
# We previously have generated ground_truth, but also simulated Nanopore reads and cut these reads into pieces. We now match the right label to each generated kmer.
#
# Notes : be careful with DNA strand

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
parser = argparse.ArgumentParser(description="Label kmers")

# path to kmers df input
parser.add_argument("-i", "--kmerInput",  required=False, type=str, default = '.Lou/Data/Output/kmers_generator/example_labelled_kmer_df.csv', help = "path for kmers dataframe, default example")

# path to output
parser.add_argument("-o", "--outputPath",  required=False, type=str, default = './', help = "path for output, default = './' ")

# path ground truth df
parser.add_argument("-g", "--groundTruth",  required=False, type=str, default = './lou/Annotation_Project/01_Part1_genic-intergenic/Output/Ground_truth_df.csv', help = "path for ground_truth df file ")

# kmer length
parser.add_argument("-m", "--mapping",  required=False, type=int, default = 'lou/Annotation_Project/Output/SepK2_mapping_df.csv', help = "path mapping file ")

# commit parser
args = parser.parse_args()

# parameters
ground_truth_file =args.groundTruth
kmers_file = args.kmerInput
mapping_file = args.mapping
path_to_output = args.outputPath

# load ground truth
ground_truth = pd.read_csv(ground_truth_file)

#load generated kmers from Nanopore reads
kmers_df = pd.read_csv(kmers_file)
kmers_df.drop(['Unnamed: 0'],axis =1, inplace = True)

# load Nanopore mappping.paf
mapping_df = pd.read_csv(mapping_file)

#Define column names
kmers_df.columns = ['Read_id_run', 'Embedding', 'Kmers', 'Pos_in_read', 'Start_in_read','Read_id']
kmers_df['Read_id']= kmers_df['Read_id'].str[1:]
kmers_df['Read_name'] = kmers_df.Read_id.str.split().str[-6]
kmers_df.drop(['Read_id'],axis = 1, inplace = True)

# if seq names unique the merge is possible
kmers_labels_df = kmers_df.merge(mapping_df, left_on= kmers_df.Read_name, right_on = mapping_df.Seq_name).drop(['key_0','Seq_name','Unnamed: 0','Start','End','Target_seq_len','Align_block_len','Nb_residue_match'], axis =1)

#get kmer lenght
kmers_labels_df['kmer_len'] = kmers_labels_df['Kmers'].apply(len)


#adjust start and end in target for kmer
kmers_labels_df['Target_start'] = kmers_labels_df['Start_in_read']+kmers_labels_df['Target_start']
kmers_labels_df['Target_end'] = kmers_labels_df.kmer_len+kmers_labels_df['Target_start']
kmers_labels_df['Genic'] = False

for i in kmers_labels_df.index:
    print('progress: ', i,'/',len(kmers_labels_df.index))
    my_kmer = kmers_labels_df.loc[i]
    # check if same strain
    target = ground_truth[ground_truth['Strain_id'] == my_kmer.Target_seq_name]
    # check if same strand
    target = target[target['strand'] == my_kmer.Strand]
    #check if kmer_start in between CDS start and end in target
    target = target[(target['start']<= my_kmer.Target_start)&(target['end']>= my_kmer.Target_start)]
    #check if something found
    if (target.empty == False):
        kmers_labels_df.Genic.loc[i] = True

#convert Embedding to numeric
kmers_labels_df.Embedding  = kmers_labels_df.Embedding.apply(lambda y : pd.to_numeric(y.strip('[]').split()))

#save output
kmers_labels_df.to_csv(path_to_output+'Labelled_kmers_df.csv')
