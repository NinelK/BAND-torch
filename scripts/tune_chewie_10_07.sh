model=kl1_gauss_bd
fac_dim=100
co_dim=4
bw=0.1 # band bw

dataset=chewie_10_07
n_all=207
n_m1=70
n_pmd=137
T=101


# just run a biRNN decoder from predicted rates
python scripts/band_performance_biRNN.py pbt-band-paper-slurm ${model} ${dataset} lfads_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} 0.
python scripts/band_performance_biRNN.py pbt-band-paper-slurm ${model} ${dataset} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
