model=kl1_gauss_chewie_10_07_cv0_pbtt
fac_dim=8
co_dim=4
bw=0.1 # band bw

dataset=chewie_10_07
n_all=207
n_m1=70
n_pmd=137
T=101

for fold in 3 4 #0 1 2 3 4
do
    python scripts/run_single.py ${model} ${dataset}_cv${fold} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
    python scripts/ablate_controls.py band-paper ${model} ${dataset}_cv${fold} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
    python scripts/band_performance.py band-paper ${model} ${dataset}_cv${fold} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
done