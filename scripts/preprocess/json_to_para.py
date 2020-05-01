import sys
import os
import requests
import re,csv
import json
import numpy as np
from bs4 import BeautifulSoup
from collections import OrderedDict
import inflect
inflect = inflect.engine()
import re, datetime
import pandas as pd
import argparse
import random


def is_date(string):
	match = re.search('\d{4}-\d{2}-\d{2}', string)
	if match:
		date = datetime.datetime.strptime(match.group(), '%Y-%m-%d').date()
		return True
	else:
		return False

def write_csv(data,split):
    with open(args['save_dir']+split+'.tsv', 'at') as outfile:
        writer = csv.writer(outfile,delimiter='\t')
        writer.writerow(data)


def config(parser):
    parser.add_argument('--json_dir', default="./../../data/tables/json/", type=str)
    parser.add_argument('--data_dir', default="./../../data/infotabs_tsv/", type=str)
    parser.add_argument('--save_dir', default="./../../temp/parapremise", type=str)
    parser.add_argument('--splits',default=["train","dev","test_alpha1","test_alpha2","test_alpha3"],  action='store', type=str, nargs='*')
    parser.add_argument('--rand_prem', default=0, type=int)
    #parser.add_argument('--multi_gpu_on', action='store_true')
    return parser

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser = config(parser)
	args = vars(parser.parse_args())
	
	for split in args["splits"]:
		data = pd.read_csv(args['data_dir']+"infotabs_"+split+".tsv",sep="\t")

		with open(args['save_dir']+split+".tsv", 'wt') as out:
			writer = csv.writer(out, delimiter='\t')
			writer.writerow(["index","table_id","annotator_id","premise","hypothesis","label"])

		if args['rand_prem'] == 2:
			table_ids = []
			for index,row in data.iterrows():
				table_ids += [row['table_id']]
			random.shuffle(table_ids)
			for index,row in data.iterrows():
				row['table_id'] = table_ids[index]

		if args['rand_prem'] == 1:
			table_ids = []
			for index,row in data.iterrows():
				table_ids += [row['table_id']]

			set_of_orignal = list(set(table_ids))
			set_of_random = set_of_orignal
			random.shuffle(set_of_random)
			set_of_orignal = list(set(table_ids))
			random_mapping_tableids = {}
			jindex = 0

			for key in set_of_orignal:
				random_mapping_tableids[key] = set_of_random[jindex]
				jindex += 1

			for index,row in data.iterrows():
				table_id = row['table_id']
				row['table_id'] = random_mapping_tableids[table_id]

		for index,row in data.iterrows():
			file = args['json_dir'] +str(row['table_id'])+".json"
			json_file = open(file,"r")
			data = json.load(json_file,object_pairs_hook=OrderedDict)
			try:
				title = data["title"][0]
			except KeyError:
				print(row)
				exit()

			del data["title"]

			para = ""

			for key in data:
				line = ""
				values = data[key]


				if (len(values) > 1) or (inflect.plural_noun(key)):
					verb_use = "are"

					if is_date("".join(values)):
						para += title+" was "+ str(key) +" on "
						line += title+" was "+ str(key) +" on "
					else:
						try:
							para += "The "+str(key)+" of "+title+" "+verb_use+" "
							line += "The "+str(key)+" of "+title+" "+verb_use+" "
						except TypeError:
							print(row)
							print(key)
							print(title)
							exit()

					for value in values[:-1]:
						para += value +", "
						line += value +", "
					if len(values) > 1:
						para += "and "+values[-1] + ". "
						line += "and "+values[-1] + ". "
					else:
						para += values[-1] + ". "
						line += values[-1] + ". "
				else:
					verb_use = "is"
					if is_date(values[0]):
						para += title+" was "+ str(key) +" on "+values[0] +". "
						line += title+" was "+ str(key) +" on "+values[0] +". "
					else:
						para +="The "+str(key)+" of "+title+" "+verb_use+" "+values[0] +". "
						line +="The "+str(key)+" of "+title+" "+verb_use+" "+values[0] +". "

			label = row["label"]
			if row["label"] == "E":
				label = 0
			if row["label"] == "N":
				label = 1
			if row["label"] == "C":
				label = 2
			
			data = [index,row['table_id'],row['annotater_id'],para,row["hypothesis"],label]
			write_csv(data,split)
