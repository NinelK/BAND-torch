model=longitudinal_data1_multisession
fac_dim=16
co_dim=0
bw=0.1 # band bw

dataset=longitudinal_data1_multisession
n=${fac_dim} # number of encoding dimensions (the ones initialized with PCs) = fac_dim
T=30

python scripts/run_pbt_longitudinal.py ${model} ${dataset} lfads_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n} 0.
python scripts/ablate_controls.py ${model} ${dataset} lfads_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n} 0.
python scripts/band_performance.py ${model} ${dataset} lfads_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n} 0.

python scripts/run_pbt_longitudinal.py ${model} ${dataset} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n} ${bw}
python scripts/ablate_controls.py ${model} ${dataset} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n} ${bw}
python scripts/band_performance.py ${model} ${dataset} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n} ${bw}
