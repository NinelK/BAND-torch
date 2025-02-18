model=multitask
fac_dim=20
co_dim=4
bw=0.01 # band bw

dataset=multitasks_tgt_onset_20140725
#n=${fac_dim} # number of encoding dimensions (the ones initialized with PCs) = fac_dim
n=75
T=50 # number of timepoints

PROJECT_STR=pbt-multitasks

# python scripts/run_pbt.py ${model} ${dataset} lfads_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n} 0.
# python scripts/ablate_controls.py ${PROJECT_STR} ${model} ${dataset} lfads_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n} 0.
# python scripts/band_performance.py ${PROJECT_STR} ${model} ${dataset} lfads_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n} 0.

# python scripts/run_pbt.py ${model} ${dataset} band_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n} ${bw}
# python scripts/ablate_controls.py ${PROJECT_STR} ${model} ${dataset} band_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n} ${bw}
# python scripts/band_performance.py ${PROJECT_STR} ${model} ${dataset} band_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n} ${bw}

model=multitask_fixed_lag

# python scripts/run_pbt.py ${model} ${dataset} band_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n} ${bw}
# python scripts/ablate_controls.py ${PROJECT_STR} ${model} ${dataset} band_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n} ${bw}
python scripts/band_performance.py ${PROJECT_STR} ${model} ${dataset} band_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n} ${bw}
