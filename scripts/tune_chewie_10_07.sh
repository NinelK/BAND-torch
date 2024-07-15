model=kl1_gauss_pbtt_r2
fac_dim=8
co_dim=4
bw=0.1 # band bw

dataset=chewie_10_07
n_all=207
n_m1=70
n_pmd=137
T=101

for fold in 0 1 2 3 4
do
    # python scripts/run_pbt.py ${model} ${dataset}_cv${fold} lfads_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} 0.
    # python scripts/ablate_controls.py pbt-band-paper ${model} ${dataset}_cv${fold} lfads_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} 0.
    # python scripts/band_performance.py pbt-band-paper ${model} ${dataset}_cv${fold} lfads_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} 0.

    python scripts/run_pbt.py ${model} ${dataset}_cv${fold} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
    python scripts/ablate_controls.py pbt-band-paper ${model} ${dataset}_cv${fold} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
    python scripts/band_performance.py pbt-band-paper ${model} ${dataset}_cv${fold} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
done