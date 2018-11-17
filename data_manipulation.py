                                                                   # data_manipulation.py
# 11th Nov. 2018
# Arnav Ghosh

import csv
import numpy as np
import os
import pickle

# CONSTANTS
POSITIVE = "+"
NEGATIVE = "-"


BASES = ["A", "C", "G", "T"]
NUM_CLASSES = 2 #O-IDX : -, 1-IDX: +

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


# align the reads first and then after that try to get contigous ones
def combine_counts_classification(class_fname, read_fname):
	with open("combined_read_counts_classes2.csv", 'w') as csvFile:
		combinedWriter = csv.writer(csvFile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
		with open(read_fname, 'r') as readFile:
			with open(class_fname, 'r') as classFile:
				read_line = readFile.readline().strip().split()

				for line in classFile:
					if read_line == []:
						break
					line = line.split()
					# cl1 ... cl2
					#			  rl ... r2 --> do nothing
					if int(line[2]) < int(read_line[1]):
						pass
					# 			  cl1 ... cl2
					# rl ... r2				--> increment readline (while this isn't true)
					elif int(line[1]) > int(read_line[2]):
						while (int(line[1]) > int(read_line[2])):
							read_line = readFile.readline().strip().split()

						# cl1 ... cl2
						#	   rl ... r2

						#     cl1 ... cl2
						# rl ... r2			--> increment until first cond.
						count = 0
						while int(read_line[1]) <= int(line[2]):
							count += int(read_line[3])
							read_line = readFile.readline().strip().split()

							if read_line == []:
								break

						line.append(count)
						combinedWriter.writerow(line)

def create_data_split(file_path, bins_per_sample, train_split=0.8, val_split=0.1, test_split=0.1):
	with open(file_path, 'r') as f:
		all_data = f.read().split('\n')
		combined_data = [all_data[i : i + bins_per_sample] for i in range(0, len(all_data), bins_per_sample)]
		np.random.shuffle(combined_data)

		len_data = len(combined_data)
		train_end_idx = int(train_split * len_data)
		val_end_idx = train_end_idx + int(val_split * len_data)

		train_set = combined_data[ : train_end_idx]
		val_set = combined_data[train_end_idx : val_end_idx]
		test_set = combined_data[val_end_idx : ]

		# TODO Not using actual base info in chromosome
		train_x, train_y = vectorize_data(train_set)
		val_x, val_y = vectorize_data(val_set)
		test_x, test_y = vectorize_data(test_set)

	return (train_x, train_y, val_x, val_y, test_x, test_y)

''' Returns X, Y where X[i] is the ith datapoint (num_samples, seq. length, 1)
					   Y[i] is a one hot representation for the ith label (num_samples, seq. length, 1)

	data: list of list of bins where each inner list forms a bin sequence
		  eg: [ ["chr1 0  24 10 +", "chr1 25 49 13 -"], -- forms --> X : [10, 13], Y [[0, 1], [1,0]]
		  		["chr1 50 74 17 +", "chr1 75 99 6  +"] ] '''
def vectorize_data(data):
	num_samples = len(data)
	bins_per_sample = len(data[0]) #should be uniform
	X = np.zeros((num_samples, bins_per_sample, 1))
	Y = np.zeros((num_samples, bins_per_sample, NUM_CLASSES))

	for i, seq in enumerate(data):
		assert len(seq) == bins_per_sample #ensure the sequence length is consistent
		X[i, ] = np.array(list(map(lambda x : int(x.split()[3]), seq))).reshape(-1, 1)
		Y[i, ] = np.array(list(map(lambda x : [0, 1] if x.split()[4] == POSITIVE else [1, 0], seq))) #assuming 2 classes

	return X, Y

# TODO: Use Chr Base Summaries
# def vectorize_aug_data(data):
