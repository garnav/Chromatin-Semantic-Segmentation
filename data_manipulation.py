# data_manipulation.py
# 11th Nov. 2018
# Arnav Ghosh

import os
import pickle

# CONSTANTS
POSITIVE = "+"
NEGATIVE = "-"
BASES = ["A", "C", "G", "T"]

#FILENAMES
DATA = "data"
PROCESSED_DATA = os.path.join(DATA, "processed_data.pickle")

'''Reads the fasta file and outputs the sequence to analyze.
Arguments:
	filename: name of the fasta file
Returns:
	s: string with relevant sequence
'''
def read_fasta(filename):
	with open(filename, "r") as f:
		s = f.read()
		newline_index = s.find("\n")
		s = s[newline_index + 1 : ]
		s = s.replace("\n", "").strip()
		return s

# chr1	3170	3194	U2	0	-
def align_histone_genome(filename, genome):
	positive_locs = []
	negative_locs = []
	positive_snps = []
	negative_snps = []

	with open(filename, "r")as f:
		for line in f:
			L = line.strip().split()
			start, end = int(L[1]) - 1, int(L[2]) - 1
			state = L[5]
			snp = genome[start : end + 1]

			if state == POSITIVE:
				positive_snps.append(snp)
				positive_locs.append((start, end))
			elif state == NEGATIVE:
				negative_snps.append(snp)
				negative_locs.append((start, end))

	return positive_locs, negative_locs, positive_snps, negative_snps

def create_data(fasta_fname, align_fname):
	genome = read_fasta(fasta_fname)
	all_data = align_histone_genome(align_fname, genome)

	with open(PROCESSED_DATA, 'wb') as f:
		pickle.dump(all_data, f)

def load_data():
	with open(PROCESSED_DATA, 'rb') as f:
		positive_locs, negative_locs, positive_snps, negative_snps = pickle.load(f)

	return positive_locs, negative_locs, positive_snps, negative_snps

# Questions
# - What is the length of data we're expecting (how many bins)
# - Two Choices
# 	- Single sequence
# 	- Binned Sequence ONLY HAVE THIS THOUGH
# Read More about
# 	- What bins represent (how are their summary statistics calculated)
#	- What to do about non-contiguos bins (How does the HMM use them?)
#	- What is U01, U02, U0
