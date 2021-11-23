##############################################
# RoBERTa Base Classifer for single sentence #
# and sentence pair classification			 #
##############################################
import argparse
import json
import os
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from transformers import AutoModel

from feedforward import FeedForward


def config(parser):
    parser.add_argument('--epochs',  default=10, type=int)
    parser.add_argument('--batch_size',  default=4, type=int)
    parser.add_argument('--in_dir', default="./../../temp/processed/parapremise/", type=str)
    parser.add_argument('--nooflabels', default=3, type=int)
    parser.add_argument('--save_dir', default="./../../temp/models", type=str)
    parser.add_argument('--save_folder', default="parapremise/", type=str)
    parser.add_argument('--model_dir',default="./../../temp/models/parapremise/",type=str)
    parser.add_argument('--model_name',default="model_4_0.301",type=str)
    parser.add_argument('--mode', default="train", type=str)
    #parser.add_argument('--embed_size',default=768,type=int)   #Obsolete
    parser.add_argument('--model_type', default='roberta-base', type=str, help='A string which determines which model will be used for training/evaluating. The value should be one of the classes mentioned on the Huggingface transformers website - https://huggingface.co/transformers/pretrained_models.html')
    parser.add_argument('--save_enable',default=0,type=int)
    parser.add_argument('--eval_splits',default=["train","dev","test_alpha1"],  action='store', type=str, nargs='*')
    parser.add_argument('--seed',default=13,type=int)
    parser.add_argument('--parallel',default=0,type=int)
    parser.add_argument('--inoculate',default=0,type=int)
    #parser.add_argument('--multi_gpu_on', action='store_true')   # use for multi gpu training (not needed)
    return parser


def test(model, classifier, data):
    """ Evaluate the model on a given dataset.
    Inputs
    ---------------
    model - transformers.AutoModel. The transformer model being used.
    classifier - torch.nn.Module. The classifier which sits on top of 
                    the transformer model
    data - dict. Consists the processed input data

    Outputs
    ---------------
    accuracy - float. Accuracy of the model on that evaluation split
    gold_inds - List[int]. Gold labels
    predictions_ind - List[int]. Parallel list to gold_inds. Contains 
                        label predictions
    """
    # Separate the data fields in the evaluation data 
    enc = torch.tensor(data['encodings']).cuda()
    attention_mask = torch.tensor(data['attention_mask']).cuda()
    segs = torch.tensor(data['segments']).cuda()
    labs = torch.tensor(data['labels']).cuda()
    ids = torch.tensor(data['uid']).cuda()

    # Create Data Loader for the split
    dataset = TensorDataset(enc,attention_mask,segs,labs,ids)
    loader = DataLoader(dataset, batch_size=args['batch_size'])

    model.eval()
    correct = 0
    total = 0 
    gold_inds = []
    predictions_inds = []

    for batch_ndx,(enc, mask, seg, gold, ids) in enumerate(loader):
	# Forward-pass w/o calculating gradients
        with torch.no_grad():
            outputs = model(enc,attention_mask = mask, token_type_ids=seg)
            predictions = classifier(outputs[1])

        # Calculate metrics
        _ , inds = torch.max(predictions,1)
        gold_inds.extend(gold.tolist())
        predictions_inds.extend(inds.tolist())
        correct+= inds.eq(gold.view_as(inds)).cpu().sum().item()
        total += len(enc)
    
    accuracy = correct/total
    
    return accuracy, gold_inds, predictions_inds



def train(args):
    """ Train the transformer model on given data
    Inputs
    -------------
    args - dict. Arguments passed via CLI
    """
    if args['inoculate']:
        train_data_file = open(args['in_dir']+"train_inoculation.pkl",'rb')
    else:
        train_data_file = open(args['in_dir']+"train.pkl",'rb')
    dev_data_file = open(args['in_dir']+"dev.pkl",'rb')
    test_data_file = open(args['in_dir']+"test_alpha1.pkl",'rb')

    # Load processed data
    train_data = pickle.load(train_data_file)
    dev_data = pickle.load(dev_data_file)
    test_data = pickle.load(test_data_file)
    
    # Creating required save directories
    if not os.path.isdir(args['save_dir']+args['save_folder']):
        os.mkdir(args['save_dir']+args['save_folder'])
        
    print("{} train data loaded".format(len(train_data['encodings'])))
    print("{} dev data loaded".format(len(dev_data['encodings'])))
    print("{} test data loaded".format(len(test_data['encodings'])))
        
    train_data_file.close()
    dev_data_file.close()
    test_data_file.close()
        
    # Separating the data fields
    train_enc = torch.tensor(train_data['encodings']).cuda()
    train_attention_mask = torch.tensor(train_data['attention_mask']).cuda()
    train_segs = torch.tensor(train_data['segments']).cuda()
    train_labs = torch.tensor(train_data['labels']).cuda()
    train_ids = torch.tensor(train_data['uid']).cuda()


    # Intialize Models
    model = AutoModel.from_pretrained(args['model_type']).cuda()
    args['embed_size'] = model.config.hidden_size
    classifier = FeedForward(args['embed_size'],int(args['embed_size']/2),args['nooflabels']).cuda()

    # Creating the training dataloaders
    dataset = TensorDataset(train_enc,train_attention_mask,train_segs,train_labs,train_ids)
    loader = DataLoader(dataset, batch_size=args['batch_size'])

    # Intialize the optimizer and loss functions
    params = list(model.parameters()) + list(classifier.parameters())
    optimizer = optim.Adagrad(params, lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()
        
    for ep in range(args['epochs']):
        epoch_loss=0
        start = time.time()
        model.train()
        # Iterate over batches
        for batch_ndx,(enc, mask, seg, gold,ids) in enumerate(tqdm(loader)):
            batch_loss = 0
            optimizer.zero_grad()
            # Forward-pass
            outputs = model(enc,attention_mask = mask, token_type_ids=seg)
            predictions = classifier(outputs[1])
            # Loss calculation and weight update
            out_loss = loss_fn(predictions,gold)
            out_loss.backward()
            optimizer.step()
                
        batch_loss += out_loss.item()
        epoch_loss+=batch_loss
                
        normalized_epoch_loss = epoch_loss/(len(loader))
        print("Epoch {}".format(ep+1))
        print("Epoch loss: {} ".format(normalized_epoch_loss))

        # Evaluate on the dev and test sets
        dev_acc, dev_gold, dev_pred = test(model,classifier,dev_data)
        test_acc, test_gold, test_pred = test(model,classifier,test_data)
        end = time.time()
        print("Dev Accuracy: {}".format(dev_acc))
        print("Time taken: {} seconds\n".format(end-start))
        # Save model
        torch.save({
                    'epoch': ep+1,
                    'model_state_dict': model.state_dict(),
                    'classifier_state_dict': classifier.state_dict(),
                    'loss': normalized_epoch_loss,
                    'dev_accuracy': dev_acc}, 
                    args["save_dir"]+args["save_folder"]+"model_"+str(ep+1)+"_"+str(dev_acc))
            
        dev_results = {"accuracy": dev_acc, "gold":dev_gold, "pred": dev_pred}
        test_results={"accuracy": test_acc, "gold":test_gold, "pred": test_pred}	
            
        with open(args["save_dir"]+args["save_folder"]+"scores_"+str(ep+1)+"_dev.json", 'w') as fp:
            json.dump(dev_results, fp)
                
        with open(args["save_dir"]+args["save_folder"]+"scores_"+str(ep+1)+"_test.json", 'w') as fp:
            json.dump(test_results, fp)



def test_data(args):
    """ Test pre-trained model on evaluation splits
    Inputs
    ----------
    args - dict. Arguments passed via CLI
    """
    result_dir = args["save_dir"]+args["save_folder"]
    # Intialize model
    model = AutoModel.from_pretrained(args['model_type']).cuda()
    args['embed_size'] = model.config.hidden_size
    classifier = FeedForward(args['embed_size'],int(args['embed_size']/2),args['nooflabels']).cuda()

    # Load pre-trained models
    checkpoint = torch.load(args['model_dir']+args['model_name'])
    model.load_state_dict(checkpoint['model_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])

    # Evaluate over splits
    for split in args["eval_splits"]:
        try:
            # Load data
            data_file = open(args['in_dir']+split+".pkl",'rb')
            data = pickle.load(data_file)
	    
            # Compute Accuracy
            acc,gold,pred = test(model,classifier,data)
            print("{} accuracy: {}".format(split, acc))
            
            results = {"accuracy": acc,"gold": gold, "pred": pred}
            
            # Save results
            if args['save_enable']!=0:
                if not os.path.isdir(result_dir):
                    os.mkdir(result_dir)
                with open(result_dir+"/predict_"+split+".json", 'w') as fp:
                    json.dump(results, fp)
        
        except FileNotFoundError:
            print("{}.pkl file doesn't exist".format(split))





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = config(parser)
    args = vars(parser.parse_args())

    current_seed = int(args['seed'])
    torch.manual_seed(current_seed)

    if args['mode']=="train":
    	train(args)
    elif args['mode'] == "test":
    	test_data(args)

	
