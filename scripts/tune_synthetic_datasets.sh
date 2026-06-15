project='pbt-lorenz'
model=lorenz
fac_dim=8
co_dim=2
causal=True

dataset=lorenz_feedback_10
n_all=50
T=100

python scripts/run_pbt.py ${project} ${model} ${dataset} lfads_causal_${fac_dim}f_${dataset} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
python scripts/ablate_controls.py ${project} ${model} ${dataset} lfads_causal_${fac_dim}f_${dataset} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
python scripts/band_performance.py ${project} ${model} ${dataset} lfads_causal_${fac_dim}f_${dataset} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}

python scripts/run_pbt.py ${project} ${model} ${dataset} band_causal_${fac_dim}f_${dataset} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
python scripts/ablate_controls.py ${project} ${model} ${dataset} band_causal_${fac_dim}f_${dataset} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
python scripts/band_performance.py ${project} ${model} ${dataset} band_causal_${fac_dim}f_${dataset} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}


causal=False
python scripts/run_pbt.py ${project} ${model} ${dataset} lfads_acausal_${fac_dim}f_${dataset} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
python scripts/ablate_controls.py ${project} ${model} ${dataset} lfads_acausal_${fac_dim}f_${dataset} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
python scripts/band_performance.py ${project} ${model} ${dataset} lfads_acausal_${fac_dim}f_${dataset} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}

python scripts/run_pbt.py ${project} ${model} ${dataset} band_acausal_${fac_dim}f_${dataset} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
python scripts/ablate_controls.py ${project} ${model} ${dataset} band_acausal_${fac_dim}f_${dataset} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
python scripts/band_performance.py ${project} ${model} ${dataset} band_acausal_${fac_dim}f_${dataset} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}