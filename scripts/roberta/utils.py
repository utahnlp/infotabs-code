##########################
# Utility functions      #
# THIS FILE IS OBSOLETE  #
##########################

import torch
import time, sys
from transformers import RobertaTokenizer, BertTokenizer

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-cased')


def _truncate_seq_pair(tokens_a, max_length):
    """Truncates a sequence pair in place to the maximum length.
    Copyed from https://github.com/huggingface/pytorch-pretrained-BERT
    """
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a)
        if total_length <= max_length:
            break
        tokens_a.pop()




def get_BERT_vector(sent1,sent2=None,max_sent1_len=400,max_sent2_len=100,single_sentence=False): 
	# Preparing attention mask so no attention is given to
	# padded tokens for sentence 1
	attention_mask_sent1 = [1]*(max_sent1_len+2)

	sent1_encoding = tokenizer.encode("<s>" + sent1,add_special_tokens=False)
	
	_truncate_seq_pair(sent1_encoding,max_sent1_len+1)
	
	attention_mask_sent1[len(sent1_encoding):max_sent1_len+1] = [0]*(max_sent1_len-len(sent1_encoding)+1)
	sent1_encoding.extend([tokenizer.encode("<pad>",add_special_tokens=False)[0]]*(max_sent1_len-len(sent1_encoding)+1)) 
	sent1_encoding.extend(tokenizer.encode("</s>",add_special_tokens=False))

	# Preparing attention mask so no attention is given tokens
	# padded tokens for sentence 2
	if not single_sentence:	
		attention_mask_sent2 = [1]*(max_sent2_len+2)

		sent2_encoding = tokenizer.encode("</s>"+sent2, add_special_tokens=False)

		attention_mask_sent2[len(sent2_encoding):max_sent2_len+1] = [0]*(max_sent2_len-len(sent2_encoding)+1)
		sent2_encoding.extend([tokenizer.encode("<pad>",add_special_tokens=False)[0]]*(max_sent2_len-len(sent2_encoding)+1))
		sent2_encoding.extend(tokenizer.encode("</s>",add_special_tokens=False))
	else:
		attention_mask_sent2 = [0]*(max_sent2_len+2)
		sent2_encoding = [tokenizer.encode("<pad>",add_special_tokens=False)[0]]*(max_sent2_len+2)

		

	# Fixing segment ids 
	segments = [0]*(max_sent1_len+2)
	if not single_sentence:
		segments.extend([1]*(max_sent2_len+2))
	else:
		segments.extend([0]*(max_sent2_len+2))



	sentences_encoding = sent1_encoding
	attention_mask = attention_mask_sent1

	sentences_encoding.extend(sent2_encoding)
	attention_mask.extend(attention_mask_sent2)


	return sentences_encoding, attention_mask, segments




def get_BERTbase_vector(sent1,sent2=None,max_sent1_len=400,max_sent2_len=100,single_sentence=False): 
	# Preparing attention mask so no attention is given to
	# padded tokens for sentence 1
	attention_mask_sent1 = [1]*(max_sent1_len+2)

	sent1_encoding = tokenizer_bert.encode("[CLS] " + sent1, add_special_tokens=False)
	
	_truncate_seq_pair(sent1_encoding,max_sent1_len+1)
	
	attention_mask_sent1[len(sent1_encoding):max_sent1_len+1] = [0]*(max_sent1_len-len(sent1_encoding)+1)
	sent1_encoding.extend([tokenizer_bert.encode("[PAD]",add_special_tokens=False)[0]]*(max_sent1_len-len(sent1_encoding)+1)) 
	sent1_encoding.extend(tokenizer_bert.encode("[SEP]",add_special_tokens=False))

	# Preparing attention mask so no attention is given tokens
	# padded tokens for sentence 2
	if not single_sentence:	
		attention_mask_sent2 = [1]*(max_sent2_len+1)

		sent2_encoding = tokenizer_bert.encode(sent2,add_special_tokens=False)

		attention_mask_sent2[len(sent2_encoding):max_sent2_len] = [0]*(max_sent2_len-len(sent2_encoding))
		sent2_encoding.extend([tokenizer_bert.encode("[PAD]",add_special_tokens=False)[0]]*(max_sent2_len-len(sent2_encoding)))
		sent2_encoding.extend(tokenizer_bert.encode("[SEP]",add_special_tokens=False))
	else:
		attention_mask_sent2 = [0]*(max_sent2_len+1)
		sent2_encoding = [tokenizer_bert.encode("[PAD]",add_special_tokens=False)[0]]*(max_sent2_len+1)

		

	# Fixing segment ids 
	segments = [0]*(max_sent1_len+2)
	if not single_sentence:
		segments.extend([1]*(max_sent2_len+1))
	else:
		segments.extend([0]*(max_sent2_len+1))



	sentences_encoding = sent1_encoding
	attention_mask = attention_mask_sent1

	sentences_encoding.extend(sent2_encoding)
	attention_mask.extend(attention_mask_sent2)


	return sentences_encoding, attention_mask, segments
