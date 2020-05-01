import pandas as pd
import json 
from collections import OrderedDict
import numpy as np
from collections import Counter
from nltk.corpus import stopwords 

stop_words = set(stopwords.words('english')) 
data_dir = "./../../data/maindata/"
table_dir = "./../../data/tables/"

def get_TableIDs(filename):
	ids = []
	data = pd.read_csv(data_dir+"infotabs_"+filename+".tsv",sep="\t",encoding="ISO-8859-1")

	for index, row in data.iterrows():
		if row['table_id'] not in ids:
			ids.append(row['table_id'])
		
	return ids 

def get_unique_keys(table_ids):
	key_set = set()
	for index in table_ids:
		with open(table_dir+"json/"+index+".json") as json_file:
			dat = json.load(json_file,object_pairs_hook=OrderedDict)
		for key in dat.keys():
			key_set.add(key.lower())
	return key_set

def get_hypolength_per_label(filename):
	label_dict = {'E':[],'N':[],'C':[]}
	data = pd.read_csv(data_dir+"infotabs_"+filename+".tsv",sep="\t",encoding="ISO-8859-1")

	for index, row in data.iterrows():
		label_dict[str(row["label"])].append(len(row["hypothesis"].split()))

	return label_dict 

def get_annotIDs(filename):
	annotator_ids = []
	annotator_dict = {}
	data = pd.read_csv(data_dir+"infotabs_"+filename+".tsv",sep="\t",encoding="ISO-8859-1")

	for index, row in data.iterrows():

		if row['annotater_id'] not in annotator_ids:
			annotator_ids.append(row['annotater_id'])
			annotator_dict[row['annotater_id']] = 1
		else:
			annotator_dict[row['annotater_id']] += 1
	
	return annotator_ids, annotator_dict

def get_table_len(table_ids):
	len_table = []
	for index in table_ids:
		with open(table_dir+"json/"+index+".json") as json_file:
			dat = json.load(json_file,object_pairs_hook=OrderedDict)
		len_table.append(len(dat.keys())-1)

	return len_table 

def unique_keys(splits):
	ids_table = {}

	for split in splits:
		ids_table[split] = get_TableIDs(split)

	unique_keys = {}

	print ("================= Total Table Keys in each split statitsics  ====================\n")
	for split in splits:
		unique_keys[split] = get_unique_keys(ids_table[split])
		print("Keys in {}: {}".format(split,len(unique_keys[split])))

	print ("\n================= Table Unique Keys statitsics  ====================\n")
	total_unique_keys = set()
	for split in splits:
		total_unique_keys = total_unique_keys.union(unique_keys[split])

	print("Total unique keys: {}".format(len(total_unique_keys)))
	print ("\n=============================================================\n\n\n")

def hypo_words(splits):
	hypothesis_per_lab = {}
	labels = {'E':"entailment",'N':"neutral",'C':"contradiction"}
	print ("================= Mean Words in hypothesis statitsics  ====================\n")
	for split in splits:
		hypothesis_per_lab[split] = get_hypolength_per_label(split)

		print("\n{}\n".format(split))
		for i in ['E','N','C']:
			print("Mean number of words for {}: {}".format(labels[i],np.mean(np.array(hypothesis_per_lab[split][i]))))
			print("Std. dev for {}: {}".format(labels[i],np.std(np.array(hypothesis_per_lab[split][i]))))
			print ("------------------------------------------")
	print ("\n=============================================================\n\n\n")

def annotators_overlap(splits):
	annotators = {}
	annotators_instances = {}


	print ("================== Annotators in each splits statistics ===================\n")
	for split in splits:
		annotators[split], annotators_instances[split] = get_annotIDs(split)
		print("Distinct annotators in {}: {}".format(split,len(annotators[split])))
		print("Annotators with #example they annotate for {}: {}".format(split,annotators_instances[split]))
		print ("------------------------------------------")

	print ("\n================= Annoatotors intersection in each splits statistics  ====================\n")
	for split in splits:
		if split=="train":
			continue
		intersection = set(annotators["train"]).intersection(annotators[split])
		print("Number of annotator intersection between train and {}: {}".format(split,len(intersection)))
		print("Annotator intersection between train and {}: {}".format(split,intersection))
		instance_overlap = 0
		for worker in list(intersection):
			instance_overlap += annotators_instances[split][worker]
		print("Instances annotated by worker who also annotated train for {}: {}".format(split, instance_overlap))
		print ("------------------------------------------")
	print ("\n=============================================================\n\n\n")

def avg_key_table(splits):
	ids_table = {}

	for split in splits:
		ids_table[split] = get_TableIDs(split)

	len_tables={}
	print ("================== Average number of keys in tables per splits statistics ===================\n")
	for split in splits:
		len_tables[split] = get_table_len(ids_table[split])
		print("Average number of keys per table in {}: {}".format(split, np.mean(np.array(len_tables[split]))))
	print ("\n=============================================================\n\n\n")

def key_intersection(splits):
	ids_table = {}

	for split in splits:
		ids_table[split] = get_TableIDs(split)

	unique_keys = {}

	for split in splits:
		unique_keys[split] = get_unique_keys(ids_table[split])
	
	print ("================= Table keys intersection in tables with train for each splits statistics  ====================\n")
	for split in splits:
		if split=="train":
			continue
		print("Intersection between Train and {}: {}".format(split, len(unique_keys["train"].intersection(unique_keys[split]))))
	print ("\n=============================================================\n\n\n")

def table_categories(splits):
	ids_table = {}

	for split in splits:
		ids_table[split] = get_TableIDs(split)

	data = pd.read_csv(table_dir+"table_categories.tsv",sep="\t",encoding="ISO-8859-1")

	ids_categories = {}
	for index, row in data.iterrows():
		ids_categories[row['table_id']] = row["category"]

	splits_categories = {}
	for split in splits:
		for table in ids_table[split]:
			try:
				splits_categories[split] += [ids_categories[table]]
			except:
				splits_categories[split] = [ids_categories[table]]

	print ("================= Table category count for each splits  ====================\n")
	for split in splits:
		print ("------------------for split : "+split+ "------------------------\n")
		count = Counter(splits_categories[split])
		# print (split, Counter(splits_categories[split]))
		print ("format 'category': (#examples, percent of #examples)")
		percentages = {k : (v, round(v*100 / float(len(splits_categories[split])),4)) for k,v in count.items()}
		print (split, percentages)
		# print ("\n------------------------------------------\n")
	print ("\n=============================================================\n\n\n")

def hypothesis_table_overlap(splits):
	ids_table = {}

	for split in splits:
		ids_table[split] = get_TableIDs(split)

	premise_tokens = {}
	for split in splits:
		for index in ids_table[split]:
			tableid = index
			with open(table_dir+"json/"+index+".json") as json_file:
				dat = json.load(json_file,object_pairs_hook=OrderedDict)
			premise_tokens[tableid] = []
			for key in dat.keys():
				premise_row = key + " "
				for value in dat[key]:
					premise_row += value +" "
				premise_row = premise_row.lower()
				premise_tokens[tableid] += set(premise_row.split())

	print ("================= Average premise hypothesis overlap for each splits  ====================\n")
	for split in splits:
		hypothesis_premise_overlap = {}
		labels = {'E':"entailment",'N':"neutral",'C':"contradiction"}

		for key in labels:
			hypothesis_premise_overlap[key] = []

		data = pd.read_csv(data_dir+"infotabs_"+split+".tsv",sep="\t",encoding="ISO-8859-1")

		for index, row in data.iterrows():
			tableid = row['table_id']
			hypothese_tokens = set((row["hypothesis"].lower()).split())
			updated_hypotoken = set([w for w in hypothese_tokens if not w in stop_words]) 

			premise_token = premise_tokens[tableid]
			intersection_set = updated_hypotoken.intersection(premise_token)
			hypothesis_premise_overlap[str(row["label"])] += [float(len(intersection_set))/len(updated_hypotoken)]
		
		print ("------------------for split : "+split+ " with following format: (label, mean, variance) ------------------------\n")
		for key in labels:
			print (labels[key], np.mean(hypothesis_premise_overlap[key]),np.std(hypothesis_premise_overlap[key]))
		print ("\n------------------------------------------\n")
	print ("\n=============================================================\n\n\n")


def table_counts(splits):
	print ("================= Table Count Statistics  ====================\n")
	ids_table = {}

	for split in splits:
		ids_table[split] = get_TableIDs(split)

	for split in splits:
		print (split,len(ids_table[split]))
	print ("\n=============================================================\n\n\n")



if __name__ == "__main__":
	splits = ["train","dev","test_alpha1","test_alpha2","test_alpha3"]
	annotators_overlap(splits)
	avg_key_table(splits)
	hypo_words(splits)
	unique_keys(splits)
	key_intersection(splits)
	table_categories(splits)
	hypothesis_table_overlap(splits)
	table_counts(splits)
