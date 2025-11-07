project='band-paper-slurm'
model=kl1_gauss_bd_d20_acausal_ci
fac_dim=100
co_dim=4
causal=False

dataset=chewie_10_07_mov
n_all=207
T=99

for fold in 0 1 2 3 4
do
python scripts/run_pbt.py ${model} ${dataset}_cv${fold} lfads_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
python scripts/ablate_controls.py ${project} ${model} ${dataset}_cv${fold} lfads_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
python scripts/band_performance.py ${project} ${model} ${dataset}_cv${fold} lfads_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}

python scripts/run_pbt.py ${model} ${dataset}_cv${fold} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
python scripts/ablate_controls.py ${project} ${model} ${dataset}_cv${fold} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
python scripts/band_performance.py ${project} ${model} ${dataset}_cv${fold} band_both_${fac_dim}f_${model} ${T} ${fac_dim} ${co_dim} ${n_all} ${causal}
done
