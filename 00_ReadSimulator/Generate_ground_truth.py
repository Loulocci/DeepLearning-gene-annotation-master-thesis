#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import seaborn as sns
from scipy import stats
import locale
locale.setlocale(locale.LC_ALL, '')

#Biopython
from Bio import SeqIO
from Bio.Seq import Seq

#my modules
from modules import Annotation_project_functions as func

#figure package
from dna_features_viewer import GraphicFeature, GraphicRecord


# ### PART 1 : Convert genbank.gb into dataframe

#Specify path to data. Should be a directory containing only .gb files of interest and nothing else.
path_to_data = './lou/Annotation_Project/00_data/00_4Ecoli_genbank/'
path_to_fig = './lou/Annotation_Project/03_Generate_kmer/Figures/'
path_to_output = './lou/Annotation_Project/03_Generate_kmer/Output/'

#get all the gb files in the directory
files = []
for f in os.listdir(path_to_data):
    if f.endswith('.gb') :
        files.append(path_to_data+f)
    else:
        print('Non genbank file found : ', f)

#Create the dataframe containing each strain present in the provided folder
Strains = []
for i in range(len(files)):
    print('file ',i,' = ',files[i])
    Strains.append(func.features_to_dataframe(files[i]))

    # loop getting the Strain id
    for line in open(files[i], 'r'):
        if re.search('VERSION', line):
            id = line.split()
            id = id[1]
            break

    Strains[i]['Strain_id'] = id

All_strains_df = pd.concat(Strains)
All_strains_df['seq_len'] = All_strains_df['end'] - All_strains_df['start']

# print example of dataframe sample
print('\n\nHere is a random sample from the final genbank Dataframe :\n\n')
print(All_strains_df.sample(1))
# print unique sequences type
print('\n\nAll the possible types of sequence :\n\n', All_strains_df['type'].unique(),'\n\n')


# ### PART 2 :convert mapping.paf into dataframe

path_to_paf = './lou/Annotation_Project/01_DeepSimulator/DeepSimulator-master/Sep_deepSim_output/Ecoli_all-v3-K2/'

#get all the .paf files in the directory
files = []
for f in os.listdir(path_to_paf):
    if f.endswith('.paf') :
        files.append(path_to_paf+f)
    else:
        print('Non .paf file found : ', f)

#Create the dataframe containing each mapping for each strain present in the provided folder
mapping = []
for i in range(len(files)):
    print('file ',i,' = ',files[i])
    mapping.append(func.paf_to_dataframe(files[i]))

mapping_df = pd.concat(mapping)


# In[35]:


print('\n\nHere is a random sample from the mapping Dataframe :\n\n')
print(mapping_df.sample(1))


# In[36]:


# change '+ and -' into 1 or -1
mapping_df.Strand[mapping_df.Strand == '+'] = 1
mapping_df.Strand[mapping_df.Strand == '-'] = -1


# In[37]:

# Print final info and stats about the ground truth
print('\n\nSome information about mapping file: generated with miniMap2 => Minimap2 identifies matching sequences in the original sequences.  It can find several matches for the same read.')
print('Mapping average length : ', mean(mapping_df.Seq_len))
print('Mapping max length : ', max(mapping_df.Seq_len))
print('Mapping min length : ', min(mapping_df.Seq_len))
print('Mapping average nb of matching residue: ', mean(mapping_df.Nb_residue_match))
print('Mapping average quality: ', mean(mapping_df.Mapping_quality))
print('Total nb of mapping :', len(mapping_df))
print('Nb of match duplicated at least once', len(mapping_df.Seq_name[mapping_df.Seq_name.duplicated()].unique()),'\n\n')


# ### PART 3 : Ground truth

# In[10]:


Ground_truth_df = All_strains_df[All_strains_df['type'] == 'CDS']

# In[11]:


# Simple gene plot to visualize a ground truth sample
EcoliUMN = Ground_truth_df[Ground_truth_df['Strain_id'] == 'NC_011751.1']
features = []
for i in range(35) :
    features.append(GraphicFeature(start=EcoliUMN['start'].iloc[i], end=EcoliUMN['end'].iloc[i], strand=EcoliUMN['strand'].iloc[i], color="#ffd700",
                   label="CDS"))

record = GraphicRecord(sequence_length=40000, features=features)
ax, _ = record.plot(figure_width=18)

#save figure
ax.figure.savefig(path_to_fig+'35CDS_EcoliUMN.png',bbox_inches='tight')


# In[12]:

# Save ground truth as csv file
Ground_truth_df= Ground_truth_df.set_index('Strain_id')
Ground_truth_df.to_csv(path_to_output+'Ground_truth_df.csv')
mapping_df.to_csv(path_to_output+'SepK2_mapping_df.csv')
