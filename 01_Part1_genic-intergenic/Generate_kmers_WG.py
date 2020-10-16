#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy import stats
import locale
locale.setlocale(locale.LC_ALL, '')
import datetime
import random
import numpy as np

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
parser.add_argument("-i", "--genomes",  required=True, type=str, help = "genome files")

# path to output
parser.add_argument("-o", "--outputPath",  required=False, type=str, default = './', help = "path for output, default = './' ")

# path to dna2vec embeddings
parser.add_argument("-e", "--embeddings",  required=False, type=str, default = './lou/Annotation_Project/01_Part1_genic-intergenic/02_embedding/dna2vec-master/results/dna2vec-20200504-1300-k10to10-100d-10c-205Mbp-sliding-oBT-rev-WG.w2v', help = "path for dna2vec embeddings, default = 10 mers embeddings ")

# path ground truth df
parser.add_argument("-g", "--groundTruth",  required=False, type=str, default = './lou/Annotation_Project/01_Part1_genic-intergenic/03_Generate_kmer/Output/Ground_truth_df.csv', help = "path for ground_truth df file ")

# kmer length
parser.add_argument("-k", "--kmerLen",  required=False, type=int, default = 7, help = "Kmer length, default = 7 ")

# commit parser
args = parser.parse_args()

# setting arguments
ground_truth_file = args.groundTruth
path_to_output = args.outputPath
reads_file = args.genomes
embedding_file = args.embeddings
kmer_len = args.kmerLen

# load ground truth
ground_truth = pd.read_csv(ground_truth_file)

#load dna2vec
model = wk.load_word2vec_format(embedding_file, binary = False)

### Prepare dataframe
print('Starting fastq file preprocessing\n\n')
genomes = pd.read_csv(genomes_file, sep = "\n", header = None)
genomes = genomes.rename(columns={0: "Seq"})

# Prepare data: match strand and strain id
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

# print a sample
genomes.sample(2)

# add read length to the df
genomes['Read_len'] = 0
for i in genomes.index:
    genomes['Read_len'][i] = len(genomes.Seq[i])

# prepare data : find nuc position
genomes['Read_start_pos'] = 0
genomes['Read_end_pos'] = 0

# iterate on each strand of each strain and create whole genome dataframe
for strain in genomes.Strain.unique():
    for strand in genomes.Strand.unique():
        read_start = []
        read_end = []
        sub_df = genomes[(genomes.Strain == strain) & (genomes.Strand == strand)]
        if sub_df.Strand.unique() == 1: # find kmer index for forward strand
            print('Forward strand indexing starts for', sub_df.Strain.unique())
            for i,seq in enumerate(sub_df.Seq):
                if i == 0:
                    read_start.append(0)
                    read_end.append(len(genomes.Seq[0])-1)
                else:
                    read_len = len(seq)
                    read_start.append(read_end[-1]+1)
                    read_end.append(read_end[-1]+read_len)
        else : # find kmer index for reverse strand
            print('Reverse strand indexing starts for', sub_df.Strain.unique())
            for i,seq in enumerate(sub_df.Seq):
                if i == 0:
                    len_genome = sub_df.Read_len.sum()-1
                    read_start.append(len_genome) # length of the genome (index is end extremity, cause reverse)
                    read_end.append(len_genome - len(seq) +1)
                else:
                    read_len = len(seq)
                    read_start.append(read_end[-1]-1)
                    read_end.append(read_end[-1]-read_len)


        genomes['Read_start_pos'].loc[(genomes.Strain == strain) & (genomes.Strand == strand)] = read_start
        genomes['Read_end_pos'].loc[(genomes.Strain == strain) & (genomes.Strand == strand)] = read_end


# then generate kmers from whole genomes
kmers = func.Multiseq_to_vector(genomes.Seq, model, kmer_size = kmer_len, rand_kmer_size = False,  shift = 1)
kmers = pd.merge(kmers, genomes[['Strand','Strain','Read_start_pos','Read_end_pos']], left_on='Read_id', right_index=True, how='left')
# get the right index for both strands
kmers['Kmer_pos_in_genome']= np.nan
kmers['Kmer_pos_in_genome'].loc[kmers.Strand == 1] = kmers.Read_start_pos + kmers.Pos_in_read
kmers['Kmer_pos_in_genome'].loc[kmers.Strand == -1] = kmers.Read_start_pos - kmers.Pos_in_read

# find protein tag corresponding to kmer
kmers['Genic'] = np.nan
kmers['Locus_tag'] = np.nan

for i in kmers.index:
    print('Step:',i,'/',len(kmers.index)-1)
    my_kmer = kmers.loc[i]
    # check if same strain
    target = ground_truth[ground_truth['Strain_id'] == my_kmer.Strain]
    # check if same strand
    target = target[target['strand'] == my_kmer.Strand]
    #check if kmer_start in between CDS start and end in target

    target = target[((target['start']<= my_kmer.Kmer_pos_in_genome)&(target['end']>= my_kmer.Kmer_pos_in_genome)) | ((target['start']<= my_kmer.Kmer_pos_in_genome+7)&(target['end']>= my_kmer.Kmer_pos_in_genome+7))]

    if (target.empty == False):
        if len(target) ==1 :
            kmers['Genic'].loc[i] = True
            kmers['Locus_tag'].loc[i] = ground_truth.locus_tag.loc[target.index.item()]
        else:
            locus =str(ground_truth.locus_tag.loc[target.index[0]])
            for row in target.index[1:]:
                locus+=', '+ground_truth.locus_tag.loc[row]


    else:
        kmers['Genic'].loc[i] = False



#save output
now = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
kmers_df.to_csv(path_to_output+'WholeReadKmers_WG_'+str(kmer_len)+'mers_'+str(now)+'_labelled')

# print info
print('Percentage of coding regions in the',len(kmers.Read_id.unique()),'reads dataset:',(len(kmers[kmers.Genic == True])/len(kmers))*100)
random_read = random.randrange(0,max(kmers.Read_id.unique()),1)
print('Here is a first example: read composition for the random read',random_read,':')
print('Strain:',kmers.Strain[(kmers.Read_id == random_read)].unique().item(),', Strand:',kmers.Strand[(kmers.Read_id == random_read)].unique().item(),', Locus found on that read:')
print(kmers.Locus_tag[(kmers.Read_id == random_read) & (kmers.Genic == True)].unique())

random_read = random.randrange(0,max(kmers.Read_id.unique()),1)
random_read =1
print('\nHere is a second example: read composition for the random read',random_read,':')
print('Strain:',kmers.Strain[(kmers.Read_id == random_read)].unique().item(),', Strand:',kmers.Strand[(kmers.Read_id == random_read)].unique().item(),', Locus found on that read:')
print(kmers.Locus_tag[(kmers.Read_id == random_read) & (kmers.Genic == True)].unique())
