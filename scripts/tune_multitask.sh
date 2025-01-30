model=multitask_noPCR
fac_dim=20
co_dim=4
bw=0.01 # band bw

dataset=multitasks_20140725
n=${fac_dim} # number of encoding dimensions (the ones initialized with PCs) = fac_dim
T=40 # number of timepoints

python scripts/run_pbt.py ${model} ${dataset} lfads_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n} 0.
python scripts/ablate_controls.py ${model} ${dataset} lfads_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n} 0.
python scripts/band_performance.py ${model} ${dataset} lfads_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n} 0.

python scripts/run_pbt.py ${model} ${dataset} band_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n} ${bw}
python scripts/ablate_controls.py ${model} ${dataset} band_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n} ${bw}
python scripts/band_performance.py ${model} ${dataset} band_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n} ${bw}
