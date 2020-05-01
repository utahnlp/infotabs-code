import os
import sys
import json
import editdistance

splits = ["dev","test_alpha3"]
reasoning_dir = "./../../data/reasoning/"

for split in splits:
	possible_reasoning = []
	fp = open(reasoning_dir+"infotabs_"+split+".tsv","r")
	lines = fp.readlines()

	reasoning_counter = {}
	number_reasoning_examples = {}

	for j in range(1,6):
		for label in ['E','C','N']:
			number_reasoning_examples[(label,j)] = 0
	for i in range(1,161):
		example = lines[i].strip("\n").split("\t")
		conseses_labels = example[4].split(",")
		label = example[3]
		number_reasoning_examples[(label,len(conseses_labels))] += 1
		for item in conseses_labels:
			try:
				reasoning_counter[(label,item)] += 1
			except:
				possible_reasoning += [item]
				reasoning_counter[(label,item)] = 1

	print ("========== reasoning for "+ split+" set ==============")
	print ("\t\tE\tN\tC")
	for reasoning in set(possible_reasoning):
		toprint = reasoning+"\t"
		for label in ['E','N','C']:
			try:
				value = reasoning_counter[(label,reasoning)]
			except:
				value = 0
			toprint += str(value)+"\t"
		print (toprint)

	print ("================================\n\n")
	print ("========== #reasoning/per_example for "+ split+" set  =========")
	toprint = "\t"
	for i in range(1,6):
		toprint += str(i)+"\t"
	print (toprint)
	for label in ['E','N','C']:
		toprint = label+"\t"
		for i in range(1,6):
			toprint += str(number_reasoning_examples[(label,i)]) +"\t"
		print (toprint)
	print ("==================================")

	print ("================================\n\n")
	print ("========== reasoning with #prediction/#gold for "+ split+" set  =========")
	fp.close()
	fp = open(reasoning_dir+"infotabs_"+split+".tsv","r")
	lines = fp.readlines()
	with open('./../../temp/models/reasoning/predict_'+ split+ '.json') as f:
		results = json.load(f)
	f.close()
	predictions = results['pred']

	j = 0
	for pred in predictions:
		if pred == 0:
			predictions[j] = "E"
		if pred == 1:
			predictions[j] = "N"
		if pred == 2:
			predictions[j] = "C"
		j += 1

	predicted_reasoning_counter = {}
	predicted_number_reasoning_examples = {}
	for j in range(1,6):
		for label in ['E','N','C']:
			predicted_number_reasoning_examples[(label,j)] = 0
	for i in range(1,161):
		example = lines[i].strip("\n").split("\t")
		conseses_labels = example[4].split(",")
		gold_label = example[3]
		pred_label = predictions[i-1]
		if gold_label == pred_label:
			predicted_number_reasoning_examples[(gold_label,len(conseses_labels))] += 1
		for item in conseses_labels:
			if gold_label == pred_label:
				try:
					predicted_reasoning_counter[(gold_label,item)] += 1
				except:
					predicted_reasoning_counter[(gold_label,item)] = 1

	print ("\t\tE\tN\tC")
	for reasoning in set(possible_reasoning):
		toprint = reasoning+"\t"
		for label in ['E','N','C']:
			try:
				value1 = reasoning_counter[(label,reasoning)]
			except:
				value1 = 0
			try:
				value2 = predicted_reasoning_counter[(label,reasoning)]
			except:
				value2 = 0
			toprint += str(value2) + "/" + str(value1)+"\t"
			# toprint += str(value2)+"\t"
		print (toprint)

	print ("================================\n\n")
	print ("========== predicted #reasoning/per_example for "+ split+" set  =========")
	toprint = "\t"
	for i in range(1,6):
		toprint += str(i)+"\t"
	print (toprint)
	for label in ['E','N','C']:
		toprint = label+"\t"
		for i in range(1,6):
			toprint += str(predicted_number_reasoning_examples[(label,i)]) +"/"+ str(number_reasoning_examples[(label,i)]) +"\t"
			# toprint += str(predicted_number_reasoning_examples[(label,i)]) +"\t"
		print (toprint)
	print ("==================================")
	
