# usr/bin/bash
#creating essential temporary directory

mkdir ./../../temp/data/parapremise
#para as a premise
python3 json_to_para.py --json_dir ./../../data/tables/json/ --data_dir ./../../data/maindata/ --save_dir ./../../temp/data/parapremise/
#structure as a premise
mkdir ./../../temp/data/strucpremise
python3 json_to_struc.py --json_dir ./../../data/tables/json/ --data_dir ./../../data/maindata/ --save_dir ./../../temp/data/strucpremise/
#wmd top 3 sentences as premise
mkdir ./../../temp/data/wmdpremise3
python3 json_to_wmd.py --json_dir ./../../data/tables/json/ --data_dir ./../../data/maindata/ --save_dir ./../../temp/data/wmdpremise3/ --topk 3
#wmd top 1 sentences as premise
mkdir ./../../temp/data/wmdpremise1
python3 json_to_wmd.py --json_dir ./../../data/tables/json/ --data_dir ./../../data/maindata/ --save_dir ./../../temp/data/wmdpremise1/ --topk 1
#random table as premise
mkdir ./../../temp/data/parapremiserand
python3 json_to_para.py --json_dir ./../../data/tables/json/ --data_dir ./../../data/maindata/ --save_dir ./../../temp/data/parapremiserand/ --rand_prem 1
#random table structure as a premise
mkdir ./../../temp/data/strucpremiserand
python3 json_to_struc.py --json_dir ./../../data/tables/json/ --data_dir ./../../data/maindata/ --save_dir ./../../temp/data/strucpremiserand/ --rand_prem 1
#dummy premise
mkdir ./../../temp/data/dummypremise
python3 json_to_dummy.py --json_dir ./../../data/tables/json/ --data_dir ./../../data/maindata/ --save_dir ./../../temp/data/dummypremise/
#reasoning dev and alpha3 using parapremise
mkdir ./../../temp/data/reasoning
python3 json_to_para.py --json_dir ./../../data/tables/json/ --data_dir ./../../data/reasoning/ --save_dir ./../../temp/data/reasoning/  --splits dev test_alpha3
