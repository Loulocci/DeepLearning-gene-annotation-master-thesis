# DeepLearning-gene-annotation-master-thesis
Master's thesis public files 

***ABSTRACT: Gene annotation remains a key issue in genetic and genomic studies. The quality of any research in this field is highly dependent on the ability to identify and characterize genes. Since the advent of high throughput sequencing technologies, the pace of research in genetics and related fields is increasing, and with it the demand for consistently superior annotation quality. Moreover, since newly sequenced genomes are usually compared with known sequences, annotation errors spread with the growth of sequence databases. The error rate depends on the sequencing method and annotation approach. Most of the annotation tools imply sequence-homology or statistical model approaches- both highly dependent on the nature of the input data. In this project, we propose a new approach allowing to confront the current struggle in gene identification, with developing a deep learning based pipeline for long and noisy reads annotation.This work is a further step towards the annotation of whole new sequences, without referring to previously sequenced genomes.***


### PART 1 Genic/intergenic: 

- Generate random k-mers (here 7-mers) from entire E. coli genome (4 diff strains) simulated noisy reads  
- Generate random k-mers (here 7-mers) from whole E. coli genome (again the 4 diff strains) 
- Label k-mers as genic or intergenic depending on position in genome (taking noise into account for simulated reads)
- Train LSTM neural network on balanced (genic/intergenic) k-mers subsets for both whole genome k-mers and simulated read k-mers
- Eval model on testing sets


### PART 2 Start codons: 

- Generate random k-mers (here 7-mers) from specific zones (containing potential start codon :ATG,TTG, GTG) in E. coli genome (4 diff strains) simulated noisy reads 
- Generate random k-mers (here 7-mers) from specific zones (containing potential start codon :ATG,TTG, GTG) in whole E. coli genome (again the 4 diff strains) 
- Label k-mers as true start or not a start depending on position in genome (taking noise into account for simulated reads)
- Train LSTM neural network on balanced (true start/not a start) k-mers subsets for both whole genome k-mers and simulated read k-mers
- Eval model on testing sets


### PART 3 Stop codons: 

Repeat part 2 but for stop codons.

- Generate random k-mers (here 7-mers) from specific zones (containing potential start codon :TAG,TGA,TAA) in E. coli genome (4 diff strains) simulated noisy reads 
- Generate random k-mers (here 7-mers) from specific zones (containing potential start codon :TAG,TGA,TAA) in whole E. coli genome (again the 4 diff strains) 
- Label k-mers as true stop or not a stop depending on position in genome (taking noise into account for simulated reads)
- Train LSTM neural network on balanced (true stop/not a stop) k-mers subsets for both whole genome k-mers and simulated read k-mers
- Eval model on testing sets


### PART 4 Scoring: 

Merge the 3 predictions together (Genic/intergenic, starts and stops), in order to get a score for candidate genes. Try to extract gene position in noisy reads from these prediction scores. 
