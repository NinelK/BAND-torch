project='band-paper'
model=kl1_gauss_bd_d20_causal_ci
fac_dim=100
co_dim=4
causal=True
n_all=207
T=101

dataset=chewie_10_07_res

# python scripts/run_pbt.py ${project} ${model} ${dataset} lfads_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
# python scripts/ablate_controls.py ${project} ${model} ${dataset} lfads_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
python scripts/band_performance.py ${project} ${model} ${dataset} lfads_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}

# python scripts/run_pbt.py ${project} ${model} ${dataset} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
# python scripts/ablate_controls.py ${project} ${model} ${dataset} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
python scripts/band_performance.py ${project} ${model} ${dataset} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}


dataset=chewie_10_07_div

# python scripts/run_pbt.py ${project} ${model} ${dataset} lfads_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
# python scripts/ablate_controls.py ${project} ${model} ${dataset} lfads_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
python scripts/band_performance.py ${project} ${model} ${dataset} lfads_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}

# python scripts/run_pbt.py ${project} ${model} ${dataset} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
# python scripts/ablate_controls.py ${project} ${model} ${dataset} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
python scripts/band_performance.py ${project} ${model} ${dataset} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
