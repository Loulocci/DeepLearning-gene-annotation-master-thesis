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

#Biopython
from Bio.Align import MultipleSeqAlignment
from Bio import SeqIO
from Bio.pairwise2 import format_alignment
from Bio import pairwise2
from Bio import Entrez
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, FeatureLocation

########################################################################
#create a dataframe from fasta file containing several genomes

#INPUT: multifasta file
#OUTPUT: pandas Dataframe of the data contained in the fasta file.

def multifasta_to_df(fasta):
    #iterate on file to detecte header and seq
    fasta = open(fasta, "r")
    header = [fasta.readline()]
    genome = []
    line = fasta.readline()
    g = 0
    while line != '':
        print('New genome')
        g += 1
        seq = line.strip('\n')
        i = 0
        while not line.startswith('>') and line != '':
            #print('Step:',i)
            line = fasta.readline()
            seq = seq+line.strip('\n')
            i += 1



        genome.append(seq)
        print('genome length:', len(seq), seq[0:240])
        if line != '':
            header.append(line)

        try :
            line = fasta.readline()
        except :
            line = ''


    # prepare columns for df
    Strain = []
    Strand = []
    for strain in header:
        if ':c' in strain:
            Strain.append(strain.strip('>\n').split(':')[0])
            Strand.append(-1)
        else:
            Strain.append(strain.strip('>\n').split()[0])
            Strand.append(1)

    #create dataframe
    seq_df = pd.DataFrame()
    seq_df['Strain'] = Strain
    seq_df['Strand'] = Strand
    seq_df['Genome'] = genome

    return seq_df

#####################################
# function that simulate random read generation from whole genome
def simulate_read(seq,read_quantity):
    seq_len = len(seq)
    reads = []
    start = []
    end = []
    read_length = []

    for read in range(read_quantity):
        read_len  = random.gauss(13000, 3000) # raw values => should be in argument
        read_len = round(read_len)
        pos = random.randint(0,seq_len - read_len)
        reads.append(seq[pos:pos+read_len])
        start.append(pos)
        end.append(pos+read_len)
        read_length.append(read_len)


    reads_df = pd.DataFrame()
    reads_df['Read_seq'] = reads
    reads_df['Read_start'] = start
    reads_df['Read_end'] = end
    reads_df['Genome_len'] = seq_len
    reads_df['Read_len'] = read_length


    return reads_df

#############################
# this function takes a string as input and inserts or deletes randomly some nuc.
# the insertion is right after random pos, the deletion is made on the nuc of given random pos and "x" next nuc depending on the size
# it outputs the modified string
def insertion_deletion(seq):
    in_del_len = 0
    seq_len = len(seq)
    del_pos = random.randint(0,seq_len-1)
    in_pos = random.randint(-1,seq_len-1)


    while in_del_len == 0:
        in_del_len = round(random.gauss(0,2))

    if in_del_len < 0: #deletion
        pos = del_pos
        out_seq = seq[0:del_pos]+seq[del_pos-in_del_len:seq_len]

    if in_del_len > 0: #insertion
        insert = ''
        pos = in_pos
        for i in range(in_del_len):
            insert += random.choice('ACGT')

        out_seq = seq[0:in_pos+1]+insert+seq[in_pos+1:seq_len] # +1 to add insert after index

    return out_seq, len(out_seq)-len(seq), pos


#######################
# this functions takes a string as input and substitute a random number of nuc
# by some random nuc. It outputs the modified sequence
def substitution(seq):
    sub_len = 0
    seq_len = len(seq)
    pos = random.randint(0,seq_len-1)

    while sub_len == 0:
        sub_len = abs(round(random.gauss(0,2)))


    if seq_len-pos <= sub_len : # avoid insertion at the end of the read
        sub_len = 1

    sub = ''
    for i in range(sub_len):
        sub += random.choice('ACGT')
        out_seq = seq[0:pos]+sub+seq[pos+sub_len:seq_len]


    return out_seq, sub_len, pos


##############################
# this function combines insertion deletion and substitution to add nosie to an input
# sequence. it returns the modified seq
def add_noise(seq, max_range):
    nb_insert_del = random.randint(0,max_range)
    nb_substitution = random.randint(0,max_range)
    indel_pos = []
    indel_size = []
    subs_pos = []
    subs_size = []

    #print('Insert/del:',nb_insert_del,'Substitution:',nb_substitution)
    for i in range(nb_insert_del):
        seq, in_del_len, in_del_pos = insertion_deletion(seq)
        indel_size.append(in_del_len)
        indel_pos.append(in_del_pos)
    for i in range(nb_substitution):
        seq, sub_len, sub_pos = substitution(seq)
        subs_size.append(sub_len)
        subs_pos.append(sub_pos)
    tot_modif = sum(list(map(abs,indel_size)))+sum(subs_size)
    return seq, tot_modif,indel_size,indel_pos, subs_size, subs_pos


##################################################
# this function gives the real index of a nuc in genome depending on the noise introduced.
#INPUT: index of the nucleotide in the noisy read, list of position modified, list of size (and type depending on the signe) of modification
# OUTPUT: index in genome. index -2 means that this nuc was inserted, and actually does not exist in genome
def find_real_index(index,indel_pos, indel_size):
    for i,pos in reversed(list(enumerate(indel_pos))) :
        if index > -2 :
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
