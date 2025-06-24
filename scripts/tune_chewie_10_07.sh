model=kl1_gauss_bd
fac_dim=100
co_dim=0

dataset=chewie_10_07
n_all=207
n_m1=70
n_pmd=137
T=101

python scripts/run_pbt_defaultHP.py ${model} ${dataset} lfads_both_${fac_dim}f_${co_dim}c_baseline ${T} ${fac_dim} ${co_dim} ${n_all} False
python scripts/run_pbt.py ${model} ${dataset} lfads_both_${fac_dim}f_${co_dim}c_${model} ${T} ${fac_dim} ${co_dim} ${n_all} True

python scripts/run_pbt_defaultHP.py ${model} ${dataset} band_both_${fac_dim}f_${co_dim}c_baseline ${T} ${fac_dim} ${co_dim} ${n_all} False
python scripts/run_pbt.py ${model} ${dataset} band_both_${fac_dim}f_${co_dim}c_${model} ${T} ${fac_dim} ${co_dim} ${n_all} True

# # just run a biRNN decoder from predicted rates
# python scripts/band_performance_biRNN.py pbt-band-paper-slurm ${model} ${dataset} lfads_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} 0.
# python scripts/band_performance_biRNN.py pbt-band-paper-slurm ${model} ${dataset} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}


