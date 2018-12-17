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
REV_CLASS_MAP = [NEGATIVE, POSITIVE]

BASES = ["A", "C", "G", "T"]
NUM_CLASSES = 10 #O-IDX : -, 1-IDX: +
DIM = 10

#FILENAMES
DATA_DIR = os.path.join("data")
PROCESSED_DATA = os.path.join(DATA_DIR, "processed_data.pickle")

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
	with open("combined_read_counts_classes2.csv", 'w', newline = '') as csvFile:
		combinedWriter = csv.writer(csvFile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
		with open(read_fname, 'r') as readFile:
			with open(class_fname, 'r') as classFile:
				read_line = readFile.readline().strip().split()

				for line in classFile:
					if read_line == []:
						break
					line = line.strip().split()
					# cl1 ... cl2
					#			  rl ... r2 --> do nothing
					if int(line[2]) < int(read_line[1]):
						pass
					# 			  cl1 ... cl2
					# rl ... r2				--> increment readline (while this isn't true)
					elif int(line[1]) > int(read_line[2]):
						#print(read_line)
						while (int(line[1]) > int(read_line[2])):
							read_line = readFile.readline().strip().split()

							if read_line == []:
								break

						# cl1 ... cl2
						#	   rl ... r2

						#     cl1 ... cl2
						# rl ... r2			--> increment until first cond.
						count = 0
						while read_line != [] and int(read_line[1]) <= int(line[2]):
							count += int(read_line[3])
							read_line = readFile.readline().strip().split()

							if read_line == []:
								break

						line.append(count)
						#print(len(line))
						line = line[:3] + line[5:]
						#print(line)
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
		print("Creating Training Set ... ")
		train_x, train_y = vectorize_data(train_set, DIM)
		#return([train_set[0]])
		print("Creating Validation Set ... ")
		val_x, val_y = vectorize_data(val_set, DIM)
		print("Creating Testing Set ... ")
		test_x, test_y = vectorize_data(test_set, DIM)

	return (train_x - np.mean(train_x), train_y,
	        val_x - np.mean(val_x), val_y,
			test_x, test_y)

def vectorize_data(data, dim):
	num_samples = len(data)
	bins_per_sample = len(data[0]) #should be uniform
	#print(bins_per_sample)
	X = np.zeros((num_samples, bins_per_sample, dim))
	Y = np.zeros((num_samples, bins_per_sample, NUM_CLASSES))

	for i, seq in enumerate(data):
		#print(i, len(seq))
		if len(seq) != bins_per_sample:
			print(len(seq))
		#assert len(seq) == bins_per_sample #ensure the sequence length is consistent
		#print(seq)
		else:
			conv_lst = np.array(list(map(lambda x : list(map(lambda y : int(y.strip()), x.split())), seq)))
			#print(conv_lst.shape)

			X[i, ] = conv_lst[:, :-1]
			y = np.zeros((bins_per_sample, NUM_CLASSES))
			#print(conv_lst[:, -1] - 1)
			y[range(bins_per_sample), conv_lst[:, -1] - 1] = 1
			Y[i, ] = y

	return X, Y

def store_data(file_path):
	train_x, train_y, val_x, val_y, test_x, test_y = create_data_split(file_path, 16, train_split=0.8, val_split=0.1, test_split=0.1)
	np.save(os.path.join(DATA_DIR, "train_x"), train_x)
	np.save(os.path.join(DATA_DIR, "train_y"), train_y)
	np.save(os.path.join(DATA_DIR, "val_x"), val_x)
	np.save(os.path.join(DATA_DIR, "val_y"), val_y)
	np.save(os.path.join(DATA_DIR, "test_x"), test_x)
	np.save(os.path.join(DATA_DIR, "test_y"), test_y)

# TODO: Use Chr Base Summaries
# def vectorize_aug_data(data):
# wgEncodeBroadHmmK562HMM.txt

# using 200bp
def normalize_base_pairs(file_path, new_file_path):
	with open(file_path, 'r') as oldFile:
		with open(new_file_path, 'w') as newFile:
			for line in oldFile:
				line = line.strip().split()
				diff_bp = int(line[2]) - int(line[1])
				prev_base = line[1]
				for i in range(int(diff_bp / 200)):
					new_line = line
					new_line[1] = prev_base
					new_line[2] = str(int(prev_base) + 200)
					newFile.write(" ".join(new_line) + "\n")
					prev_base = str(int(prev_base) + 200)

def combine_data_labels(data_file_path, class_file_path):
	with open("chromHmm_data_labels_generated.bed", 'w') as outFile:
		with open(data_file_path, 'r') as data:
			print(data.readline()) #ignore header lines
			print(data.readline()) #ignore header lines
			with open(class_file_path, 'r') as label_data:
		 		# use shorter Set
				for line in label_data:
					x_data = data.readline().strip().split()
					full_label = line.split()[3]
					# old data
					# label = full_label[:full_label.find('_')]
					# new data --> generated via ChromHMM
					label = full_label[full_label.find('E') + 1 : ].strip()
					data_label = x_data + [label]
					outFile.write(" ".join(data_label) + '\n')

def augment_data_labels(data_path, base_pair_file, fasta):
	with open("chromHmm_data_labels_augmented.bed", 'w') as out_file:
		with open(base_pair_file, 'r') as base_file:
			with open(data_path, 'r') as data_file:
				for line in data_file:
					line = line.strip().split()
					bp_line = base_file.readline().strip().split()
					#print(bp_line)
					start_bp = int(bp_line[1])
					end_bp = int(bp_line[2])

					base_counts = count_bases(fasta[start_bp : end_bp])
					full_label = bp_line[3]
					label = full_label[full_label.find('E') + 1 : ].strip()

					if base_counts is not None:
						new_line = line + base_counts + [label]
						#print(new_line)
						out_file.write(" ".join(new_line) + '\n')

def count_bases(fasta):
	num_A = 0
	num_C = 0
	num_G = 0
	num_T = 0

	for base in fasta:
		#print(repr(base))
		if base == 'N':
			return None

		if base == 'A':
			#print("base")
			num_A += 1
		elif base == 'C':
			num_C += 1
		elif base == 'G':
			num_G += 1
		elif base == 'T':
			num_T += 1

	return [str(num_A), str(num_C), str(num_G), str(num_T)]

def main():
	store_data("data/chromHmm_data_labels_generated.bed")

if __name__ == '__main__':
	main()