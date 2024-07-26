model=kl1_gauss_bd
fac_dim=8
co_dim=4
bw=0.1 # band bw #TODO substitute with a binary flag

dataset=chewie_10_07
n_all=207
n_m1=70
n_pmd=137
T=101
postfix=_d20_causal_ci

for fold in 0 1 2 3 4
do
    # python scripts/run_single.py ${model} ${dataset}_cv${fold} band_both_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
    # python scripts/ablate_controls.py band-paper-slurm ${model} ${dataset}_cv${fold} band_both_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
    # python scripts/band_performance.py band-paper-slurm ${model} ${dataset}_cv${fold} band_both_${fac_dim}f_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
    
    python scripts/ablate_controls.py band-paper-slurm ${model} ${dataset}_cv${fold} band_both_${fac_dim}f_${co_dim}c_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
    python scripts/band_performance.py band-paper-slurm ${model} ${dataset}_cv${fold} band_both_${fac_dim}f_${co_dim}c_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}

    # python scripts/ablate_controls.py band-paper-slurm ${model} ${dataset}_cv${fold} lfads_both_${fac_dim}f_${co_dim}c_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} 0.
    # python scripts/band_performance.py band-paper-slurm ${model} ${dataset}_cv${fold} lfads_both_${fac_dim}f_${co_dim}c_${model}${postfix} ${T} ${fac_dim} ${co_dim} ${n_all} 0.
done

# pbt with folds

# for fold in 0 1 2 3 4
# do
#     python scripts/ablate_controls.py pbt-band-paper-slurm ${model} ${dataset}_cv${fold} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
#     python scripts/band_performance.py pbt-band-paper-slurm ${model} ${dataset}_cv${fold} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${bw}
# done

# single runs with folds