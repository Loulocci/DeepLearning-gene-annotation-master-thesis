#!/usr/bin/env python
# coding: utf-8

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
parser.add_argument("-i", "--genomes",  required=True, type=str, help = "file for reads dataframe")

# path to output
parser.add_argument("-o", "--outputPath",  required=False, type=str, default = './', help = "path for output, default = './' ")

# path to dna2vec embeddings
parser.add_argument("-e", "--embeddings",  required=False, type=str, default = '/tudelft.net/staff-umbrella/abeellab/lou/Annotation_Project/01_Part1_genic-intergenic/02_embedding/dna2vec-master/results/dna2vec-20200504-1300-k10to10-100d-10c-205Mbp-sliding-oBT-rev-WG.w2v', help = "path for dna2vec embeddings, default = 10 mers embeddings ")

# path ground truth df
parser.add_argument("-g", "--groundTruth",  required=False, type=str, default = '/tudelft.net/staff-umbrella/abeellab/lou/Annotation_Project/01_Part1_genic-intergenic/03_Generate_kmer/Output/Ground_truth_df.csv', help = "path for dna2vec embeddings, default = 4 Strains Ecoli ")

# kmer length
parser.add_argument("-k", "--kmerLen",  required=False, type=int, default = 10, help = "Kmer length, default = 7 ")

# commit parser
args = parser.parse_args()

# on server
ground_truth_file = args.groundTruth
path_to_output = args.outputPath
genomes_file = args.genomes
embedding_file = args.embeddings
kmer_len = args.kmerLen

# load ground truth
ground_truth = pd.read_csv(ground_truth_file)

#load dna2vec
model = wk.load_word2vec_format(embedding_file, binary = False)

####
#### Prepare dataframe
####

print('Starting fastq file preprocessing\n\n')
genomes = pd.read_csv(genomes_file, sep = "\n", header = None)
genomes = genomes.rename(columns={0: "Seq"})


# Prepare data: match strand and train id
strain_vec = []
strand_vec = []
for i, seq in enumerate(genomes.Seq) :

    if seq.startswith('>'):
        print('New strain found : ')
        try:
            strain = seq.split(':c')[0]
            strand = seq.split(':c')[1]
            strand = -1
        except :
            strain = seq.split()[0]
            strand = 1

        strain = strain.strip('>')
        genomes.drop([i], axis = 0, inplace = True)
        print(strain,'Strand: ',strand)
    else:
        strain_vec.append(strain)
        strand_vec.append(strand)


genomes.index = range(len(genomes))
genomes['Strain'] = strain_vec
genomes['Strand'] = strand_vec
genomes.Strain.loc[genomes.Strain == 'CP003289.1'] = 'NC_018658.1'


# prepare data : find nuc position
genomes['Read_start_pos'] = 0
genomes['Read_end_pos'] = 0

# iterate on each strand of each strain
for strain in genomes.Strain.unique():
    for strand in genomes.Strand.unique():
        read_start = []
        read_end = []
        sub_df = genomes[(genomes.Strain == strain) & (genomes.Strand == strand)]
        if sub_df.Strand.unique() == 1: # find kmer index for forward strand
            for i,seq in enumerate(sub_df.Seq):
                if i == 0:
                    read_start.append(0)
                    read_end.append(len(genomes.Seq[0])-1)
                else:
                    read_len = len(seq)
                    read_start.append(read_end[-1]+1)
                    read_end.append(read_end[-1]+read_len)
        else : # find kmer index for reverse strand
            for i,seq in enumerate(sub_df.Seq):
                if i == 0:
                    len_genome = len(seq)*len(sub_df) -1
                    read_start.append(len_genome) # length of the genome (index is end extremity cause reverse)
                    read_end.append(len_genome - len(seq) +1)
                else:
                    read_len = len(seq)
                    read_start.append(read_end[-1]-1)
                    read_end.append(read_end[-1]-read_len)

        genomes['Read_start_pos'].loc[(genomes.Strain == strain) & (genomes.Strand == strand)] = read_start
        genomes['Read_end_pos'].loc[(genomes.Strain == strain) & (genomes.Strand == strand)] = read_end

#####
##### Generate Kmers
#####

print('Starting kmer generation \n\n')
#init columns to store start codons index
genomes['Start_codon_index'] = 0
genomes['Start_codon_index'] = genomes['Start_codon_index'].astype(object)


for i, read in enumerate(genomes.Seq):
    print('STEP',i,'/',len(genomes.Seq)-1)

    ATG_index = func.get_codon_index('ATG', read)
    #genomes['ATG_index'].loc[i] = ATG_index
    GTG_index = func.get_codon_index('GTG', read)
    #genomes['GTG_index'].loc[i] = GTG_index
    TTG_index = func.get_codon_index('TTG', read)
    genomes['Start_codon_index'].loc[i] = ATG_index+GTG_index+TTG_index


codon_kmers = func.MultiReads_to_codonKmers(genomes.Seq, model, genomes.Start_codon_index, kmer_size = kmer_len)


#codon_kmers_df= pd.concat([codon_kmers,genomes.head[['Strain','Strand']]], axis=1,  join='inner')
codon_kmers_df = pd.merge(codon_kmers, genomes[['Strain','Strand','Read_start_pos']], left_on='Read_id', right_index=True, how='left')
codon_kmers_df['Kmer_pos_in_genome'] = codon_kmers_df['Read_start_pos']+codon_kmers_df['Pos_kmer_in_read']
codon_kmers_df['Codon_pos_in_genome'] = codon_kmers_df['Read_start_pos']+codon_kmers_df['Pos_codon_in_read']

####
#### Labelling ATG codons in the 4 WG Ecoli
####

start_count = 0
codon_kmers_df['Is_start_codon'] = 0

for i in codon_kmers_df.index:
    print('STEP',i,'/',len(codon_kmers_df))
    kmer = codon_kmers_df.loc[i]
    # check if same strain
    target = ground_truth[ground_truth['Strain_id'] == kmer.Strain]
    # check if same strand
    target = target[target['strand'] == kmer.Strand]
    #check if the kmer has an actual start in the target
    if kmer.Strand == 1:
        target = target[(target['start']<= kmer.Codon_pos_in_genome+1) & (target['start']>= kmer.Codon_pos_in_genome-1)]
    else:
        # the codon start is located at the biggest numerical extremity of the gene in reverse strand
        target = target[(target['end']<= kmer.Codon_pos_in_genome+1) & (target['end']>= kmer.Codon_pos_in_genome-1)]
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


#print info
print(len(codon_kmers_df[codon_kmers_df['Is_start_codon']==1]), 'start codons identified!')
print('NC_000913.3 :',len(codon_kmers_df.Codon_pos_in_genome[(codon_kmers_df['Is_start_codon'] == 1) & (codon_kmers_df['Strain']== 'NC_000913.3')].unique()))
print('NC_011751.1 :',len(codon_kmers_df.Codon_pos_in_genome[(codon_kmers_df['Is_start_codon'] == 1) & (codon_kmers_df['Strain']== 'NC_011751.1')].unique()))
print('NC_018658.1 :',len(codon_kmers_df.Codon_pos_in_genome[(codon_kmers_df['Is_start_codon'] == 1) & (codon_kmers_df['Strain']== 'NC_018658.1')].unique()))
print('NC_002695.2 :',len(codon_kmers_df.Codon_pos_in_genome[(codon_kmers_df['Is_start_codon'] == 1) & (codon_kmers_df['Strain']== 'NC_002695.2')].unique()))


#Save output
now = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
codon_kmers_df.to_csv(path_to_output+'StartCodons_'+str(kmer_len)+'mers_'+str(now))
