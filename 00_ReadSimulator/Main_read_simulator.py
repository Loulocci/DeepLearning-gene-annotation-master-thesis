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
import string

#my modules
from modules import Readsimulator_functions as func

# argument parser
parser = argparse.ArgumentParser(description="Read simulator run config")

# fasta file input
parser.add_argument("-i", "--fastafile",  required=True, type=str, help = "input fasta_file")

# output path
parser.add_argument("-o", "--output",  required=False, type=str, default = './', help = "path to output dir, default  = here =  ./ ")

# Number of read generated per sequence
parser.add_argument("-r", "--readnb",  required=False, type=int, default = 100, help = "Number of read generated per sequence. If there is more than one sequence in fasta then => read tot =  read number * nb seq. Default 100")

# Number of read generated per sequence
parser.add_argument("-n", "--noise",  required=False, type=int, default = 10, help = "Nucleotide proportion of read which are modified (deletion, substitution, insertion). Default 10 ( = 10%) ")

# commit parser
args = parser.parse_args()

#Provided in args
fasta_file = args.fastafile
path_to_output = args.output
read_quantity = args.readnb
noise_percentage = args.noise


# start processing
seq_df = func.multifasta_to_df(fasta_file) #load fastafile into dataframe

#generate reads from whole genomes
reads_df = pd.DataFrame()
for i,seq in enumerate(seq_df.Genome):
    new_reads =  func.simulate_read(seq,read_quantity)
    new_reads['Strain'] = seq_df.Strain.loc[i]
    new_reads['Strand'] = seq_df.Strand.loc[i]
    reads_df = pd.concat([reads_df,new_reads], ignore_index=True)

# reindex reverse strand to standard
reads_df.Read_start[reads_df.Strand == -1] = reads_df.Genome_len - reads_df.Read_start
reads_df.Read_end[reads_df.Strand == -1] = reads_df.Genome_len - reads_df.Read_end

# Introduce noise in reads
reads_df['Noisy_read_len'] = 0
reads_df['Noisy_read'] = 0
reads_df['Modif_percent'] = 0
for i in reads_df.index :
    seq = reads_df.Read_seq.loc[i]
    seq, tot_modif = func.add_noise(seq, round(len(seq)*noise_percentage/100))
    reads_df.Noisy_read.loc[i] = seq
    reads_df.Noisy_read_len.loc[i] = len(seq)
    reads_df.Modif_percent.loc[i] = (tot_modif/len(reads_df.Read_seq.loc[i]))*100

#Save output
now = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
reads_df.to_csv(path_to_output+'ReadSim_nb-'+str(read_quantity)+'_noise-'+str(noise_percentage)+'_date-'+str(now))
