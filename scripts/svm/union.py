import pandas as pd
import numpy as np
from numpy import linalg as LA

data_dir="./../../temp/data/parapremise/"
save_dir="./../../temp/svmformat/union"


def get_dimensions(train_data,dev_data,test_data, test_adverse_data, alpha3_data):
	bigram_to_index = {}
	index_to_bigram = []

	for index,row in train_data.iterrows():
		#label = int(row["label"])
		try:
			if row["hypothesis"][-1] == '.':
				row["hypothesis"] = row["hypothesis"][:-1]
		except:
			print(index)
			print(row["hypothesis"])
			continue
		unigrams = row["hypothesis"].split(" ")[:-1]
		bigrams = [b for b in zip(row["hypothesis"].split(" ")[:-1], row["hypothesis"].split(" ")[1:])]
		bigrams += unigrams
		bigrams = set(bigrams)
		try:
			unigrams_premise = row["premise"].split(" ")[:-1]
			bigrams_premise = [b for b in zip(row["premise"].split(" ")[:-1], row["premise"].split(" ")[1:])]
			bigrams_premise += unigrams_premise
			bigrams_premise = set(bigrams_premise)
		except:
			bigrams_premise = bigrams
		bigrams_union = list(bigrams.union(bigrams_premise))
		for b in bigrams_union:
			key = " ".join(b)
			if key not in bigram_to_index.keys():
				bigram_to_index[key] = len(index_to_bigram)
				index_to_bigram.append(key)
		if (index+1)%1000 == 0:
			print("{} examples finshed".format(index+1))

	for index,row in dev_data.iterrows():
		#label = int(row["label"])
		if row["hypothesis"][-1] == '.':
			row["hypothesis"] = row["hypothesis"][:-1]
		unigrams = row["hypothesis"].split(" ")[:-1]
		bigrams = [b for b in zip(row["hypothesis"].split(" ")[:-1], row["hypothesis"].split(" ")[1:])]
		bigrams += unigrams
		bigrams = set(bigrams)
		try:
			unigrams_premise = row["premise"].split(" ")[:-1]
			bigrams_premise = [b for b in zip(row["premise"].split(" ")[:-1], row["premise"].split(" ")[1:])]
			bigrams_premise += unigrams_premise
			bigrams_premise = set(bigrams_premise)
		except:
			bigrams_premise = bigrams
		bigrams_union = list(bigrams.union(bigrams_premise))
		for b in bigrams_union:
			key = " ".join(b)
			if key not in bigram_to_index.keys():
				bigram_to_index[key] = len(index_to_bigram)
				index_to_bigram.append(key)
		if (index+1)%1000 == 0:
			print("{} examples finshed".format(index+1))

	
	for index,row in test_data.iterrows():
		#label = int(row["label"])
		if row["hypothesis"][-1] == '.':
			row["hypothesis"] = row["hypothesis"][:-1]
		unigrams = row["hypothesis"].split(" ")[:-1]
		bigrams = [b for b in zip(row["hypothesis"].split(" ")[:-1], row["hypothesis"].split(" ")[1:])]
		bigrams += unigrams
		bigrams = set(bigrams)
		try:
			unigrams_premise = row["premise"].split(" ")[:-1]
			bigrams_premise = [b for b in zip(row["premise"].split(" ")[:-1], row["premise"].split(" ")[1:])]
			bigrams_premise += unigrams_premise
			bigrams_premise = set(bigrams_premise)
		except:
			bigrams_premise = bigrams
		bigrams_union = list(bigrams.union(bigrams_premise))
		for b in bigrams_union:
			key = " ".join(b)
			if key not in bigram_to_index.keys():
				bigram_to_index[key] = len(index_to_bigram)
				index_to_bigram.append(key)

		if (index+1)%1000 == 0:
			print("{} examples finshed".format(index+1))

	for index,row in test_adverse_data.iterrows():
		#label = int(row["label"])
		if row["hypothesis"][-1] == '.':
			row["hypothesis"] = row["hypothesis"][:-1]
		unigrams = row["hypothesis"].split(" ")[:-1]
		bigrams = [b for b in zip(row["hypothesis"].split(" ")[:-1], row["hypothesis"].split(" ")[1:])]
		bigrams += unigrams
		bigrams = set(bigrams)

		unigrams_premise = row["premise"].split(" ")[:-1]
		bigrams_premise = [b for b in zip(row["premise"].split(" ")[:-1], row["premise"].split(" ")[1:])]
		bigrams_premise += unigrams_premise
		bigrams_premise = set(bigrams_premise)

		bigrams_union = list(bigrams.union(bigrams_premise))
		for b in bigrams_union:
			key = " ".join(b)
			if key not in bigram_to_index.keys():
				bigram_to_index[key] = len(index_to_bigram)
				index_to_bigram.append(key)

		if (index+1)%1000 == 0:
			print("{} examples finshed".format(index+1))

	for index,row in alpha3_data.iterrows():
		#label = int(row["label"])
		if row["hypothesis"][-1] == '.':
			row["hypothesis"] = row["hypothesis"][:-1]
		unigrams = row["hypothesis"].split(" ")[:-1]
		bigrams = [b for b in zip(row["hypothesis"].split(" ")[:-1], row["hypothesis"].split(" ")[1:])]
		bigrams += unigrams
		bigrams = set(bigrams)

		try:
			unigrams_premise = row["premise"].split(" ")[:-1]
			bigrams_premise = [b for b in zip(row["premise"].split(" ")[:-1], row["premise"].split(" ")[1:])]
			bigrams_premise += unigrams_premise
			bigrams_premise = set(bigrams_premise)
		except:
			bigrams_premise = bigrams

		bigrams_union = list(bigrams.union(bigrams_premise))
		for b in bigrams_union:
			key = " ".join(b)
			if key not in bigram_to_index.keys():
				bigram_to_index[key] = len(index_to_bigram)
				index_to_bigram.append(key)

		if (index+1)%1000 == 0:
			print("{} examples finshed".format(index+1))

	return bigram_to_index, index_to_bigram


def get_data(data, bigram_to_index):
	trainable_data = np.zeros((1,len(bigram_to_index)))
	labels = np.array([])
	for index,row in data.iterrows():
		labels = np.append(labels,int(row["label"]))
		data_point = np.zeros((1,len(bigram_to_index)))
		if row["hypothesis"][-1] == '.':
			row["hypothesis"] = row["hypothesis"][:-1]
		unigrams = row["hypothesis"].split(" ")[:-1]
		bigrams = [b for b in zip(row["hypothesis"].split(" ")[:-1], row["hypothesis"].split(" ")[1:])]
		bigrams += unigrams
		for b in bigrams:
			key = " ".join(b)
			data_point[0][bigram_to_index[key]] = 1
		trainable_data = np.append(trainable_data,data_point,axis=0)

		if (index+1)%1000 == 0:
			print("{} examples finshed".format(index))

	trainable_data = trainable_data[1:]
	return trainable_data, labels

def get_data_svm_format(data, bigram_to_index):
	trainable_data = ""
	for index,row in data.iterrows():
		data_point = str(int(row["label"])+1) + " "
		try:
			if row["hypothesis"][-1] == '.':
				row["hypothesis"] = row["hypothesis"][:-1]
		except:
			print(index)
			print(row["hypothesis"])
			continue
		unigrams = row["hypothesis"].split(" ")[:-1]
		bigrams = [b for b in zip(row["hypothesis"].split(" ")[:-1], row["hypothesis"].split(" ")[1:])]
		bigrams += unigrams
		bigrams = set(bigrams)

		try:
			unigrams_premise = row["premise"].split(" ")[:-1]
			bigrams_premise = [b for b in zip(row["premise"].split(" ")[:-1], row["premise"].split(" ")[1:])]
			bigrams_premise += unigrams_premise
			bigrams_premise = set(bigrams_premise)
		except:
			bigrams_premise = bigrams

		bigrams_union = list(bigrams.union(bigrams_premise))
		bigram_active = []
		for b in bigrams_union:
			key = " ".join(b)
			bigram_active += [int(bigram_to_index[key])+1]
			bigram_active = list(set(bigram_active))
			bigram_active.sort()
		for b in bigram_active[:-1]:
			data_point += str(b) + ":" + "1" + " "
		try:
			trainable_data += data_point + str(bigram_active[-1]+1) +":"+"1"+"\n"
		except:
			pass
		if (index+1)%1000 == 0:
			print("{} examples finshed".format(index))

	return trainable_data

if __name__ == "__main__":
	train_data = pd.read_csv(data_dir+"train.tsv",sep="\t",encoding="ISO-8859-1")
	dev_data = pd.read_csv(data_dir+"dev.tsv",sep="\t",encoding="ISO-8859-1")
	test_data = pd.read_csv(data_dir+"test_alpha1.tsv",sep="\t",encoding="ISO-8859-1")
	test_adverse_data = pd.read_csv(data_dir+"test_alpha2.tsv",sep="\t",encoding="ISO-8859-1")
	alpha3_data = pd.read_csv(data_dir+"test_alpha3.tsv",sep="\t",encoding="ISO-8859-1")
	bigram_to_index, index_to_bigram = get_dimensions(train_data,dev_data,test_data,test_adverse_data, alpha3_data)
	print("Got_dimensions")
	#np.save("lookup.npy",np.array(index_to_bigram))
	train_data_final = get_data_svm_format(train_data,bigram_to_index)
	fp = open(save_dir+"train.txt","w+")
	fp.write(train_data_final)

	dev_data_final = get_data_svm_format(dev_data,bigram_to_index)
	fp = open(save_dir+"dev.txt","w+")
	fp.write(dev_data_final)


	test_data_final = get_data_svm_format(test_data,bigram_to_index)
	fp = open(save_dir+"test_alpha1.txt","w+")
	fp.write(test_data_final)

	test_data_adverse_final = get_data_svm_format(test_adverse_data,bigram_to_index)
	fp = open(save_dir+"test_alpha2.txt","w+")
	fp.write(test_data_adverse_final)

	alpha3_data_final = get_data_svm_format(alpha3_data,bigram_to_index)
	fp = open(save_dir+"test_alpha3.txt","w+")
	fp.write(alpha3_data_final)
