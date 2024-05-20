#!/bin/bash

# Define the parameters
dataset="chewie_10_07"
model="band"
area="M1"
batch_size=256
controls=2

if [ "$area" = "M1" ]; then
    neurons=70
elif [ "$area" = "PMd" ]; then
    neurons=137
elif [ "$area" = "all" ]; then  
    neurons=207
fi

if [ "$model" = "lfads" ]; then
    bw=0.
elif [ "$model" = "band" ]; then
    bw=0.1
fi

# Define the list of factors
factors=("100")
samples=("0" "1" "2")

# Iterate over factors
for sample_n in "${samples[@]}"; do

    for factor in "${factors[@]}"; do
        if [ "$area" = "all" ]; then  
            model_file="${dataset}"
            folder="${model}_${factor}f_${controls}c_kl1_studentT_bs${batch_size}_sample${sample_n}" 
        else
            model_file="${dataset}_${area}"
            folder="${model}_${area}_${factor}f_${controls}c_kl1_studentT_bs${batch_size}_sample${sample_n}" 
        fi
        params="${model_file} ${folder} ${factor} ${controls} ${neurons} ${bw}" 
        echo "Running ${params}"
        python scripts/run_scaling.py "$model_file" "$folder" "$factor" "$controls" "$neurons" "$bw" "$sample_n"
        python scripts/ablate_controls.py "$model_file" "$folder" "$factor" "$controls" "$neurons" "$bw" "$sample_n"
        python scripts/band_performance.py "$model_file" "$folder" "$factor" "$controls" "$neurons" "$bw" "$sample_n"
    done
done