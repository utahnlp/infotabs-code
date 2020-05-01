import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as plticker
from statistics import mode
import ast
import random
import argparse
from sklearn.metrics import cohen_kappa_score


def config(parser):
    parser.add_argument('--json_dir', default="./../../data/tables/json/", type=str)
    parser.add_argument('--input_dir', default="./../../data/validation/", type=str)
    parser.add_argument('--save_dir', default="./../../temp/validation/", type=str)
    parser.add_argument('--filename', default="metric_summary", type=str)
    parser.add_argument('--splits',default=["dev","test_alpha1","test_alpha2","test_alpha3"],  action='store', type=str, nargs='*')
    #parser.add_argument('--multi_gpu_on', action='store_true')
    return parser

def fleiss_kappa(M):
  """
  See `Fleiss' Kappa <https://en.wikipedia.org/wiki/Fleiss%27_kappa>`_.
  :param M: a matrix of shape (:attr:`N`, :attr:`k`) where `N` is the number of subjects and `k` is the number of categories into which assignments are made. `M[i, j]` represent the number of raters who assigned the `i`th subject to the `j`th category.
  :type M: numpy matrix
  """
  N, k = M.shape  # N is # of items, k is # of categories
  # n_annotators = float(np.sum(M[0, :]))  # # of annotators
  n_annotators = 5

  p = np.sum(M, axis=0) / (N * n_annotators)
  P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
  Pbar = np.sum(P) / N
  PbarE = np.sum(p * p)

  kappa = (Pbar - PbarE) / (1 - PbarE)

  return kappa

def prep_data(data):
	result = {}
	cnf_matrix = np.zeros((3,3,3))
	result["3"] = result["4"] = result["5"] = 0
	result["3_gold"] = result["4_gold"] = result["5_gold"] = 0
	result["no"] = len(data)

	individual_total_prediction = 0
	individual_agreement_with_gold = 0
	individual_total_prediction_with_majority = 0
	individual_agreement_with_majority = 0

	M = np.zeros((result["no"],3))

	most_common_list = []
	gold_common_list = []
	for index,row in data.iterrows():
		gold_label = row["first_human_label"]
		pred_labels = row["other_human_labels"]
		pred_labels = ast.literal_eval(pred_labels)
		# for mistakenly double validated tables #
		if len(pred_labels) > 5:
			random.shuffle(pred_labels)
			pred_labels = pred_labels[0:5]
		# for mistakenly double validated tables #
		most_common,num_most_common = Counter(pred_labels).most_common(1)[0]
		for prediction in pred_labels:
			M[index][prediction] += 1
			individual_total_prediction += 1
			if str(gold_label)==str(prediction):
				individual_agreement_with_gold += 1
		if num_most_common > 2:
			most_common_list += [int(most_common)]
			gold_common_list += [int(gold_label)]
			for prediction in pred_labels:
				individual_total_prediction_with_majority += 1
				if str(most_common)==str(prediction):
					individual_agreement_with_majority += 1
			result[str(num_most_common)] += 1
			if str(most_common) == str(gold_label):
				result[str(num_most_common)+"_gold"]+=1
			cnf_matrix[int(num_most_common)-3][int(gold_label)][int(most_common)] +=1
	
	cnf_matrix[0] = np.matrix(cnf_matrix[0])
	cnf_matrix[1] = np.matrix(cnf_matrix[1])
	cnf_matrix[2] = np.matrix(cnf_matrix[2])
	result["individual agreement with gold"] = float(individual_agreement_with_gold)/individual_total_prediction
	result["individual agreement with majority"] = float(individual_agreement_with_majority)/individual_total_prediction_with_majority
	result["fleiss kappa"] = fleiss_kappa(M)
	result["cohen kappa"] = cohen_kappa_score(most_common_list, gold_common_list)

	return (result,cnf_matrix)

def getPlot(result,split,output_dir):
	perf={}
	perf_gold = {}
	perf["3"] = (result["3"]+result["4"]+result["5"])/result["no"]*100
	perf_gold["3"] = (result["3_gold"]+result["4_gold"]+result["5_gold"])/result["no"]*100
	perf["4"] = (result["4"]+result["5"])/result["no"]*100
	perf_gold["4"] = (result["4_gold"]+result["5_gold"])/result["no"]*100
	perf["5"] = result["5"]/result["no"]*100
	perf_gold["5"] = result["5_gold"]/result["no"]*100
	result["majority agreement at least"] = perf
	result["majority and gold agreement at least"] = perf_gold

	plt.clf()
	plt.grid(axis='y',linestyle='--', linewidth=0.5)
	bar1  =plt.bar(range(len(perf)), list(perf.values()), align='center',label='bar1')
	bar2  =plt.bar(range(len(perf_gold)), list(perf_gold.values()), align='center',color='gold',label='bar2')
	plt.xticks(range(len(perf)), list(perf.keys()))
	plt.yticks(range(0,100,10))

	plt.xlabel("Minimum number of annotators in agreement (out of 5)")
	plt.ylabel("Percentage of Examples")
	plt.title("Human Validation Performance Pilot for "+split+" set")
	plt.legend((bar1,bar2),('Agreement within annotators','Agreement within annotators & gold label'))
	plt.savefig(output_dir+'plots/'+split+'.png')
	plt.show()
	return 1

if __name__=="__main__":

	parser = argparse.ArgumentParser()
	parser = config(parser)
	args = vars(parser.parse_args())

	splits = args["splits"]
	input_dir = args["input_dir"]
	output_dir = args["save_dir"]
	filename = args["filename"]

	fo = open(output_dir+filename+".txt", "w")
	for split in splits:
		data = pd.read_csv(input_dir+"infotabs_valid_"+split+".tsv",sep="\t")
		(result,cnf_matrix) = prep_data(data)
		pred_keys = ["3","4","5","no"]
		gold_keys = ["3_gold","4_gold","5_gold"]
		no_agreement = 0

		fo.write("--------------------------------------------------------------------"+"\n")
		fo.write("result of validation for "+split+" set"+"\n")
		getPlot(result,split,output_dir)
		fo.write("--------------------------------------------------------------------"+"\n")
		fo.write("exactly 'x' agreement \t exactly 'x' agreement and equals gold label"+"\n")
		fo.write("--------------------------------------------------------------------"+"\n")
		for i in range(0,3):
			pred_key = pred_keys[i]
			gold_key = gold_keys[i]
			fo.write(" ".join([str(pred_key),":",str(result[pred_key]),"\t",str(gold_key),":",str(result[gold_key])])+"\n")
			no_agreement += result[pred_key]

		no_agreement = result["no"] - no_agreement
		fo.write(" ".join(["no_agreement",":",str(no_agreement)])+"\n")

		fo.write("--------------------------------------------------------------------"+"\n")
		fo.write("final statistics in terms of percentage or standard evaluation metric"+"\n")
		fo.write("--------------------------------------------------------------------"+"\n")
		for key in result:
			if (key not in pred_keys) and (key not in gold_keys): 
				fo.write(" ".join([str(key),":", str(result[key])])+"\n")
		fo.write("--------------------------------------------------------------------"+"\n\n\n\n")
	fo.close()
	