##############################################
# Data Preprocessing for RoBERTa Base        #
##############################################
import argparse
import json
import os
import sys
import time
import torch

from utils import get_BERT_vector


def preprocess_config(parser):
        """ Adds Command line arguments to an empty Argument Parser
        Input
        --------------
        parser - argparse.ArgumentParser . An initialized empty Arg Parser

        Output
        --------------
        parser - argparse.ArgumentParser. Returns the argument parser with added
                arguments
        """
	parser.add_argument('--max_len', default= 512, type=int)
	parser.add_argument('--data_dir', default="./../../temp/data/", type=str)
	parser.add_argument('--in_dir', default="parapremise",type=str)
	parser.add_argument('--out_dir',default="../processed/parapremise",type=str)
	parser.add_argument('--single_sentence', default=0,type=int)
	parser.add_argument('--splits',default=["train","dev","test_alpha1","test_alpha2","test_alpha3"],  action='store', type=str, nargs='*')
	return parser



def load_sentences(file, skip_first=True, single_sentence=False):
	""" Loads sentences into process-friendly format for a given file path.
        Inputs
        -------------------
        file    - str or pathlib.Path. The file path which needs to be processed
        skip_first      - bool. If True, skips the first line.
        single_sentence - bool. If True, Only the hypothesis statement is chosen. 
                            Else, both the premise and hypothesis statements are 
                            considered. This is useful for hypothesis bias experiments.

        Outputs
        --------------------
        rows    - List[dict]. Consists of all data samples. Each data sample is a 
                    dictionary containing- uid, hypothesis, premise (except hypothesis
                    bias experiment), and the NLI label for the pair
        """
        rows = []
	with open(file, encoding="ISO-8859-1") as f:
		for line in f:
			if skip_first:
				skip_first = False
				continue
			blocks = line.strip().split('\t')   # Converts to list
			
                        # Takes the relevant elements of the row necessary. Putting them in a dict,
			lab = int(blocks[-1])
			if single_sentence:
				sample = {'uid': int(blocks[0]), 'hypothesis': blocks[4], 'label': lab}
			else:	
				sample = {'uid': int(blocks[0]), 'premise': blocks[3], 'hypothesis': blocks[4], 'label': lab}
			rows.append(sample) # Append the loaded sample
	return rows



if __name__ == "__main__":
        # Parse Command line arguments 
	parser = argparse.ArgumentParser()
	parser = preprocess_config(parser)
	args = vars(parser.parse_args())

	data = {}
        # Loads data into process-friendly format for each split
	for split in args["splits"]:
		data[split] = load_sentences(args['data_dir']+args['in_dir']+"/"+split+".tsv",
									single_sentence=args['single_sentence'])
		print("{} {} samples loaded".format(len(data[split]),split))

        # Name of the folder where processed data will be stored
	if args['single_sentence']:
		folder_name = args['data_dir']+args['out_dir']+"_hypo"
	else:
		folder_name = args['data_dir']+args['out_dir']

        # Checks if output folder exists or not
	if not os.path.isdir(folder_name):
		os.mkdir(folder_name)
	elif len(os.listdir(folder_name)) != 0:
		char = input("""\nThere are existing files in your save directory.\nThis may overwrite data if a split name coincides with a file name. Do you wish to continue? [y/n]: """)
		if char != "y" and char !="Y":
			exit()
	
        # Process for every split
	for split in args["splits"]:
                #Initialize dictionary to store processed information
                keys = ["uid","encodings","attention_mask","segments","labels"]
                data_dict = {key:[] for key in keys}
		samples_processed = 0
                # Iterate over all data poitns
		for pt_dict in data[split]:
			samples_processed += 1
			if args['single_sentence']:
				enc, amask, segments = get_BERT_vector(pt_dict['hypothesis'],single_sentence=True)
			else:
				enc, amask, segments = get_BERT_vector(pt_dict['premise'],pt_dict['hypothesis'])

			#if len(data_dict.keys())==0:
			#	data_dict['uid'] = [int(pt_dict['uid'])]
		        #	data_dict['encodings'] = [enc]
			#	data_dict['attention_mask'] = [amask]
			#	data_dict['segments'] = [segments]
			#	data_dict['labels']   = [pt_dict['label']]
			#else:
			data_dict['uid'].append(int(pt_dict['uid']))
			data_dict['encodings'].append(enc)
			data_dict['attention_mask'].append(amask)
			data_dict['segments'].append(segments)
			data_dict['labels'].append(pt_dict['label'])
			
			if (samples_processed % 100) == 0:
				print("{} examples processed".format(samples_processed))

		print("{} processing done.".format(split))
		# Dump data to pickle
		output = open(folder_name+"/"+split+".pkl", 'wb')
		pickle.dump(data_dict, output)
		output.close()

	print("Preprocessing Finished")











