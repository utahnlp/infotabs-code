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
    parser.add_argument('--save_dir', default="./../../temp/parapremise/", type=str)
    parser.add_argument('--splits',default=["train","dev","test_alpha1","test_alpha2","test_alpha3"],  action='store', type=str, nargs='*')
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

		for index,row in data.iterrows():

			label = row["label"]
			if row["label"] == "E":
				label = 0
			if row["label"] == "N":
				label = 1
			if row["label"] == "C":
				label = 2
			
			data = [index,row['table_id'],row['annotater_id'],"to be or not to be",row["hypothesis"],label]
			write_csv(data,split)
