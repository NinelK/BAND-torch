project='pbt-lorenz'
model=lorenz
fac_dim=8
co_dim=2
n_all=50
T=100

# Loop through each dataset
for dataset in lorenz_feedback_10 lorenz_0 lorenz_forward_10; do
    # Loop through causal settings
    for causal in True False; do
        
        # Determine the naming string based on the boolean value
        if [ "$causal" = "True" ]; then
            mode="causal"
        else
            mode="acausal"
        fi
        
        # Loop through the prefixes (lfads and band)
        for prefix in lfads band; do
            run_name="${prefix}_${mode}_${fac_dim}f_${dataset}"
            
            python scripts/run_pbt.py ${project} ${model} ${dataset} ${run_name} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
            python scripts/ablate_controls.py ${project} ${model} ${dataset} ${run_name} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
            python scripts/band_performance.py ${project} ${model} ${dataset} ${run_name} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
        done
    done
done