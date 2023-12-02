#! /bin/bash
echo "--Moving data and config files into temporary directory--"
source activate neurocaas
neurocaas-contrib workflow get-data
neurocaas-contrib workflow get-config

echo "--Parsing paths--"
datapath=$(neurocaas-contrib workflow get-datapath)
configpath=$(neurocaas-contrib workflow get-configpath)
resultpath=$(neurocaas-contrib workflow get-resultpath-tmp)

echo "--Running AutoLFADS--"
source activate band-torch
python /home/ubuntu/band-torch/scripts/run_pbt.py $datapath $configpath $resultpath
source deactivate

echo "--Writing results--"
cd $resultpath/best_model
zip -r autoband.zip *
neurocaas-contrib workflow put-result -r autoband.zip

source deactivate
